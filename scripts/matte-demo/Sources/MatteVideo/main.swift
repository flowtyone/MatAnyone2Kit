import AVFoundation
import CoreImage
import CoreML
import CoreVideo
import Foundation
import MatAnyoneKitCoreML

// Offline MatAnyone2 runner.
//
//   swift run MatteVideo <input.mp4> <output.mp4> [--mode cutout|alpha|composite] [--bg image.jpg]
//
// Modes:
//   cutout    – subject over solid black (default; clean, identity-safe matte demo)
//   alpha     – white silhouette on black (the pure matte; great technical shot)
//   composite – subject over a background image (pass --bg); the "teleport" look.
//               Supplying --bg implies composite.

enum Mode: String { case cutout, alpha, composite }

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

// MARK: - Args
let argv = CommandLine.arguments
guard argv.count >= 3 else {
    print("usage: MatteVideo <input.mp4> <output.mp4> [--mode cutout|alpha|composite] [--bg image.jpg]")
    exit(2)
}
let inputURL = URL(fileURLWithPath: argv[1])
let outputURL = URL(fileURLWithPath: argv[2])
var mode = Mode.cutout
var modeExplicit = false
var bgURL: URL?
do {
    var i = 3
    while i < argv.count {
        switch argv[i] {
        case "--mode":
            i += 1
            guard i < argv.count, let m = Mode(rawValue: argv[i]) else { fail("--mode needs cutout|alpha|composite") }
            mode = m; modeExplicit = true
        case "--bg":
            i += 1
            guard i < argv.count else { fail("--bg needs a path") }
            bgURL = URL(fileURLWithPath: argv[i])
        default:
            fail("unknown argument \(argv[i])")
        }
        i += 1
    }
}
if bgURL != nil && !modeExplicit { mode = .composite }

// MARK: - Setup
guard let matte = MatAnyoneMatte() else { fail("failed to load MatAnyone models") }
let ci = CIContext(options: [.cacheIntermediates: false])
let rgb = CGColorSpaceCreateDeviceRGB()
let black = CIImage(color: CIColor(red: 0, green: 0, blue: 0))
let white = CIImage(color: CIColor(red: 1, green: 1, blue: 1))

var bgImage: CIImage?
if let bgURL {
    guard let img = CIImage(contentsOf: bgURL) else { fail("could not load background \(bgURL.path)") }
    bgImage = img
}

// MARK: - Reader
let asset = AVURLAsset(url: inputURL)
guard let track = asset.tracks(withMediaType: .video).first else { fail("no video track in \(inputURL.path)") }
let reader: AVAssetReader
do { reader = try AVAssetReader(asset: asset) } catch { fail("reader: \(error.localizedDescription)") }
let trackOut = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
trackOut.alwaysCopiesSampleData = false
reader.add(trackOut)
guard reader.startReading() else { fail("startReading: \(reader.error?.localizedDescription ?? "?")") }

// MARK: - Compositing helpers
func aspectFillBackground(width: Int, height: Int) -> CIImage {
    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    guard let bgImage else { return black.cropped(to: rect) }
    let e = bgImage.extent
    let scale = max(CGFloat(width) / e.width, CGFloat(height) / e.height)
    let scaled = bgImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    let s = scaled.extent
    let tx = (CGFloat(width) - s.width) / 2 - s.origin.x
    let ty = (CGFloat(height) - s.height) / 2 - s.origin.y
    return scaled.transformed(by: CGAffineTransform(translationX: tx, y: ty)).cropped(to: rect)
}

func compose(frame: CVPixelBuffer, alpha: CVPixelBuffer?, width: Int, height: Int) -> CIImage {
    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    let src = CIImage(cvPixelBuffer: frame)

    guard let alpha else {
        // No matte yet (subject not seeded) — keep the timeline intact.
        return mode == .alpha ? black.cropped(to: rect) : src
    }

    // Alpha is OneComponent8 at 288×512; scale to output and turn luminance into an alpha channel.
    let aw = CGFloat(CVPixelBufferGetWidth(alpha)), ah = CGFloat(CVPixelBufferGetHeight(alpha))
    let maskAlpha = CIImage(cvPixelBuffer: alpha)
        .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / aw, y: CGFloat(height) / ah))
        .applyingFilter("CIMaskToAlpha")

    let fg: CIImage
    let bg: CIImage
    switch mode {
    case .alpha:     fg = white.cropped(to: rect); bg = black.cropped(to: rect)
    case .cutout:    fg = src;                     bg = black.cropped(to: rect)
    case .composite: fg = src;                     bg = aspectFillBackground(width: width, height: height)
    }

    return fg.applyingFilter("CIBlendWithMask", parameters: [
        kCIInputBackgroundImageKey: bg,
        kCIInputMaskImageKey: maskAlpha,
    ]).cropped(to: rect)
}

// MARK: - Writer (configured lazily once we know the frame size)
var writer: AVAssetWriter?
var writerInput: AVAssetWriterInput?
var adaptor: AVAssetWriterInputPixelBufferAdaptor?
var outW = 0, outH = 0

func configureWriter(width: Int, height: Int) {
    outW = width; outH = height
    try? FileManager.default.removeItem(at: outputURL)
    guard let w = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else { fail("writer init") }
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
    ])
    input.expectsMediaDataInRealTime = false
    let ad = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ])
    w.add(input)
    guard w.startWriting() else { fail("startWriting: \(w.error?.localizedDescription ?? "?")") }
    w.startSession(atSourceTime: .zero)
    writer = w; writerInput = input; adaptor = ad
}

func makeOutputBuffer() -> CVPixelBuffer {
    if let pool = adaptor?.pixelBufferPool {
        var buf: CVPixelBuffer?
        if CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buf) == kCVReturnSuccess, let buf {
            return buf
        }
    }
    var buf: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, outW, outH, kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buf)
    guard let buf else { fail("could not allocate output buffer") }
    return buf
}

// MARK: - Process
var frames = 0
let start = Date()

func process(_ sample: CMSampleBuffer) {
    guard let px = CMSampleBufferGetImageBuffer(sample) else { return }
    let pts = CMSampleBufferGetPresentationTimeStamp(sample)
    let w = CVPixelBufferGetWidth(px), h = CVPixelBufferGetHeight(px)
    if writer == nil { configureWriter(width: w, height: h) }

    var alpha: CVPixelBuffer?
    matte.matte(px) { alpha = $0.alpha }

    let image = compose(frame: px, alpha: alpha, width: outW, height: outH)
    let outBuf = makeOutputBuffer()
    CVPixelBufferLockBaseAddress(outBuf, [])
    ci.render(image, to: outBuf, bounds: CGRect(x: 0, y: 0, width: outW, height: outH), colorSpace: rgb)
    CVPixelBufferUnlockBaseAddress(outBuf, [])

    while !(writerInput?.isReadyForMoreMediaData ?? false) { usleep(2000) }
    adaptor?.append(outBuf, withPresentationTime: pts)
    frames += 1
    if frames % 15 == 0 { print("…\(frames) frames") }
}

while let sample = trackOut.copyNextSampleBuffer() {
    process(sample)
}

if reader.status == .failed { fail("read failed: \(reader.error?.localizedDescription ?? "?")") }
guard let writer, let writerInput else { fail("no frames decoded") }
writerInput.markAsFinished()
let sem = DispatchSemaphore(value: 0)
writer.finishWriting { sem.signal() }
sem.wait()
if writer.status == .failed { fail("write failed: \(writer.error?.localizedDescription ?? "?")") }

let dt = Date().timeIntervalSince(start)
print("done: \(frames) frames in \(String(format: "%.1f", dt))s → \(outputURL.path) [mode \(mode.rawValue)]")
