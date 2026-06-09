import CoreImage
import CoreVideo
import Vision

/// Camera ⟷ Core ML tensor bridge for the fixed-shape MatAnyone2 pipeline.
///
/// The exported models run at a fixed working size (`W`×`H`, e.g. 288×512 portrait), so there's no
/// dynamic scale. The camera frame is resized (plain, framing-preserving — the
/// compositor's aspect-fill UVs keep the alpha registered) into a channel-first RGB `[1,3,H,W]`
/// float tensor in [0,1], matching the PyTorch layout the models were traced in. The alpha comes
/// back as `[H*W]` and is written into a single-channel `CVPixelBuffer` at `W`×`H`.
final class CoreMLFrameBridge {
    let W: Int
    let H: Int
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    private var rgbaPool: CVPixelBufferPool?
    private var alphaPool: CVPixelBufferPool?

    init(width: Int, height: Int) {
        self.W = width
        self.H = height
        rgbaPool = Self.makePool(width: width, height: height, format: kCVPixelFormatType_32BGRA)
        alphaPool = Self.makePool(width: width, height: height,
                                  format: kCVPixelFormatType_OneComponent8)
    }

    private static func makePool(width: Int, height: Int, format: OSType) -> CVPixelBufferPool? {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
        return pool
    }

    /// Camera BGRA → channel-first RGB `[1,3,H,W]` in [0,1].
    func imageTensor(from camera: CVPixelBuffer) -> MatAnyoneCoreML.Tensor? {
        guard let rgbaPool else { return nil }
        var dst: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, rgbaPool, &dst) == kCVReturnSuccess,
              let dst else { return nil }

        let cw = CVPixelBufferGetWidth(camera), ch = CVPixelBufferGetHeight(camera)
        let img = CIImage(cvPixelBuffer: camera).transformed(
            by: CGAffineTransform(scaleX: CGFloat(W) / CGFloat(cw), y: CGFloat(H) / CGFloat(ch)))
        context.render(img, to: dst, bounds: CGRect(x: 0, y: 0, width: W, height: H),
                       colorSpace: rgbColorSpace)

        CVPixelBufferLockBaseAddress(dst, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dst, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(dst)
        let hw = H * W
        var data = [Float](repeating: 0, count: 3 * hw)
        let inv: Float = 1.0 / 255.0
        data.withUnsafeMutableBufferPointer { out in
            for y in 0..<H {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                let rb = y * W
                for x in 0..<W {
                    let p = x * 4                         // BGRA
                    out[rb + x] = Float(row[p + 2]) * inv          // R -> plane 0
                    out[hw + rb + x] = Float(row[p + 1]) * inv     // G -> plane 1
                    out[2 * hw + rb + x] = Float(row[p]) * inv     // B -> plane 2
                }
            }
        }
        return .init(data: data, shape: [1, 3, H, W])
    }

    /// alpha `[H*W]` in [0,1] → single-channel `CVPixelBuffer` (OneComponent8) at W×H.
    func alphaBuffer(from alpha: [Float]) -> CVPixelBuffer? {
        guard let alphaPool else { return nil }
        var dst: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, alphaPool, &dst) == kCVReturnSuccess,
              let dst else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }
        guard let base = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(dst)
        alpha.withUnsafeBufferPointer { src in
            for y in 0..<H {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                let rb = y * W
                for x in 0..<W {
                    row[x] = UInt8(max(0, min(255, src[rb + x] * 255 + 0.5)))
                }
            }
        }
        return dst
    }
}

/// First-frame person seed via Apple Vision, returning a binary mask `[H*W]` at the working size.
///
/// Pure CoreVideo/CoreImage/Vision (no Core ML matte model). `VNGeneratePersonSegmentationRequest` gives the combined
/// person mask; `VNDetectHumanRectanglesRequest` picks the closest (largest-box) person and we
/// intersect the two so the seed tracks one subject.
final class CoreMLPersonSeeder {
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let grayColorSpace = CGColorSpaceCreateDeviceGray()
    private var maskPool: CVPixelBufferPool?
    private var poolW = 0, poolH = 0

    /// Returns the seed mask `[H*W]` in {0,1}, or nil if no person found.
    func seed(from camera: CVPixelBuffer, width W: Int, height H: Int) -> [Float]? {
        let handler = VNImageRequestHandler(cvPixelBuffer: camera, orientation: .up, options: [:])
        let seg = VNGeneratePersonSegmentationRequest()
        seg.qualityLevel = .accurate
        seg.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let human = VNDetectHumanRectanglesRequest()

        guard (try? handler.perform([seg, human])) != nil,
              let maskBuffer = (seg.results?.first as? VNPixelBufferObservation)?.pixelBuffer,
              var mask = resizedMask(maskBuffer, w: W, h: H) else { return nil }

        // Binarize.
        var fg: Float = 0
        for i in 0..<mask.count { let v: Float = mask[i] >= 0.5 ? 1 : 0; mask[i] = v; fg += v }
        guard fg > 0 else { return nil }

        // Intersect with the closest (largest) human box, if any.
        if let closest = (human.results ?? []).max(by: {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        }) {
            let box = closest.boundingBox
            let x0 = clamp(Int((box.minX * CGFloat(W)).rounded(.down)), 0, W)
            let x1 = clamp(Int((box.maxX * CGFloat(W)).rounded(.up)), 0, W)
            let yTop = clamp(Int(((1 - box.maxY) * CGFloat(H)).rounded(.down)), 0, H)  // Vision y up
            let yBot = clamp(Int(((1 - box.minY) * CGFloat(H)).rounded(.up)), 0, H)
            var masked = [Float](repeating: 0, count: W * H)
            var keep: Float = 0
            for y in yTop..<max(yTop, yBot) {
                let row = y * W
                for x in x0..<max(x0, x1) { let v = mask[row + x]; masked[row + x] = v; keep += v }
            }
            if keep > 0 { return masked }            // else fall back to the whole-person mask
        }
        return mask
    }

    private func resizedMask(_ mask: CVPixelBuffer, w: Int, h: Int) -> [Float]? {
        if maskPool == nil || poolW != w || poolH != h {
            poolW = w; poolH = h
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_OneComponent8,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
            maskPool = pool
        }
        guard let maskPool else { return nil }
        var dst: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, maskPool, &dst) == kCVReturnSuccess,
              let dst else { return nil }

        let sw = CVPixelBufferGetWidth(mask), sh = CVPixelBufferGetHeight(mask)
        let img = CIImage(cvPixelBuffer: mask).transformed(
            by: CGAffineTransform(scaleX: CGFloat(w) / CGFloat(sw), y: CGFloat(h) / CGFloat(sh)))
        context.render(img, to: dst, bounds: CGRect(x: 0, y: 0, width: w, height: h),
                       colorSpace: grayColorSpace)

        CVPixelBufferLockBaseAddress(dst, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dst, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(dst)
        var flat = [Float](repeating: 0, count: w * h)
        let inv: Float = 1.0 / 255.0
        for y in 0..<h {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
            let rb = y * w
            for x in 0..<w { flat[rb + x] = Float(row[x]) * inv }
        }
        return flat
    }

    private func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int { max(lo, min(hi, v)) }
}
