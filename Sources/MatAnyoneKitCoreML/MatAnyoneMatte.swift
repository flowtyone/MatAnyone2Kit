import CoreML
import CoreVideo
import Foundation
import QuartzCore
import os

/// Real-time foreground matting backed by the MatAnyone2 Core ML / ANE port — the package's
/// top-level entry point.
///
/// First frame: a Vision person mask seeds the memory (closest person), then MatAnyone tracks the
/// matte frame-to-frame from its own memory (no per-frame Vision pass). Runs at the models' fixed
/// working resolution; the alpha is framed identically to the camera frame (full-frame, no
/// letterbox), so a compositor can sample it with the same aspect-fill UVs as the camera. All
/// inference happens synchronously on the calling queue.
///
/// The six precompiled models ship inside the package (`Bundle.module`), so the only thing a host
/// app supplies is camera `CVPixelBuffer`s:
///
///     let matte = MatAnyoneMatte()
///     matte?.matte(pixelBuffer) { frame in
///         // frame.alpha is a single-channel CVPixelBuffer (1 = foreground), or nil = passthrough
///     }
public final class MatAnyoneMatte {
    /// Outcome of one matte pass. `alpha` is the single-channel foreground matte (1 = subject,
    /// 0 = background) at the model working resolution but framed like the camera frame; `nil`
    /// means "no matte yet — show the full camera frame" (passthrough). `timing` carries the
    /// per-stage wall-clock for the pass (nil on passthrough and on the seed frame, whose warmup
    /// steps aren't steady-state).
    public struct MatteFrame {
        public let alpha: CVPixelBuffer?
        public let timing: Timing?

        public struct Timing {
            public var preprocessMs: Double
            public var inferenceMs: Double
            public var postprocessMs: Double
        }

        public static let passthrough = MatteFrame(alpha: nil, timing: nil)
    }

    /// Verbose per-frame diagnostics to Console (subsystem `com.elsewhere.matanyonekit`, category
    /// `matte`). Useful while bringing a device path up; set to `false` for steady-state.
    public static var diagnostics = false

    /// Per-model compute placement. Everything conv-heavy runs on the ANE — the win a pure-GPU path
    /// couldn't get. `readout`/`maskencoder` were made ANE-eligible by re-export (CAResBlock ECA
    /// rank-4 rewrite, SDPA decomposition, static slices). The `object_summarizer` (rank-5 / matmul
    /// / reduce) that broke `maskencoder`'s ANE compile is its own `objsummary` model, pinned to the
    /// GPU and run only on memory frames so the conv-only `maskencoder` hits the ANE every frame.
    public static var defaultUnit: MLComputeUnits = .cpuAndNeuralEngine
    public static var unitOverrides: [String: MLComputeUnits] = [
        "objsummary": .cpuAndGPU,
    ]

    private let engine: MatAnyoneCoreMLEngine
    private let bridge: CoreMLFrameBridge
    private let seeder = CoreMLPersonSeeder()
    private var seeded = false
    private var frameIndex = 0

    private static let log = Logger(subsystem: "com.elsewhere.matanyonekit", category: "matte")

    /// Working resolution the bundled models run at (portrait, e.g. 288×512).
    public var workingWidth: Int { bridge.W }
    public var workingHeight: Int { bridge.H }

    /// Loads the MatAnyone2 Core ML models. Pass `modelsDir` to load from a custom location;
    /// otherwise the precompiled models bundled in the package (`Bundle.module`) are used.
    /// Returns `nil` if the models are missing or unloadable.
    public init?(modelsDir: URL? = nil) {
        guard let dir = modelsDir
            ?? Bundle.module.url(forResource: "MatAnyone", withExtension: nil) else {
            Self.log.error("MatAnyone models folder not found in package bundle")
            return nil
        }
        guard let model = try? MatAnyoneCoreML(modelsDir: dir, computeUnits: Self.defaultUnit,
                                               unitOverrides: Self.unitOverrides) else {
            Self.log.error("failed to load Core ML models from \(dir.path)")
            return nil
        }
        self.engine = MatAnyoneCoreMLEngine(model: model)
        self.bridge = CoreMLFrameBridge(width: model.workingW, height: model.workingH)
        MatAnyoneCoreML.profilingEnabled = Self.diagnostics
        MemoryBank.profilingEnabled = Self.diagnostics
        let units = MatAnyoneCoreML.modelNames.map {
            "\($0):\(Self.unitOverrides[$0]?.rawValue ?? Self.defaultUnit.rawValue)"
        }
        Self.log.notice("loaded models, working \(model.workingW)x\(model.workingH), units [\(units.joined(separator: " "))]")
        if Self.diagnostics, #available(iOS 17.4, macOS 14.4, *) {
            for name in ["readout", "maskencoder", "decoder"] {
                model.logANEUnsupportedOps(name) { Self.log.notice("ANE-probe \($0)") }
            }
        }
    }

    /// Compute the foreground matte for one camera frame and deliver it via `completion` (called
    /// synchronously on the current queue). Returns `.passthrough` until a person is seeded.
    public func matte(_ pixelBuffer: CVPixelBuffer, completion: (MatteFrame) -> Void) {
        frameIndex += 1
        let diag = Self.diagnostics
        let cw = CVPixelBufferGetWidth(pixelBuffer), ch = CVPixelBufferGetHeight(pixelBuffer)

        let t0 = CACurrentMediaTime()
        guard let image = bridge.imageTensor(from: pixelBuffer) else {
            if diag { Self.log.error("frame \(self.frameIndex): imageTensor failed → passthrough") }
            completion(.passthrough); return
        }
        let t1 = CACurrentMediaTime()

        let alpha: [Float]
        let isSeedFrame: Bool
        do {
            if seeded {
                alpha = try engine.step(image: image)
                isSeedFrame = false
            } else {
                let tSeed = CACurrentMediaTime()
                guard let seed = seeder.seed(from: pixelBuffer, width: bridge.W, height: bridge.H) else {
                    if diag {
                        Self.log.notice("frame \(self.frameIndex): no person in seed (cam \(cw)x\(ch)) → passthrough, retry")
                    }
                    completion(.passthrough); return       // no person yet — retry next frame
                }
                if diag {
                    let cov = seed.reduce(0, +) / Float(seed.count)
                    Self.log.notice("frame \(self.frameIndex): SEED ok cov \(cov, format: .fixed(precision: 3)) in \((CACurrentMediaTime()-tSeed)*1000, format: .fixed(precision: 1))ms (cam \(cw)x\(ch))")
                }
                alpha = try engine.seed(image: image, seedMask: seed)
                seeded = true
                isSeedFrame = true
            }
        } catch {
            Self.log.error("frame \(self.frameIndex): engine threw \(error.localizedDescription) → passthrough (seeded=\(self.seeded))")
            completion(.passthrough); return
        }
        let t2 = CACurrentMediaTime()

        guard let buffer = bridge.alphaBuffer(from: alpha) else {
            if diag { Self.log.error("frame \(self.frameIndex): alphaBuffer failed → passthrough") }
            completion(.passthrough); return
        }
        let t3 = CACurrentMediaTime()

        if diag && (isSeedFrame || frameIndex % 15 == 0) {
            logStats(alpha: alpha, pre: (t1 - t0) * 1000, inf: (t2 - t1) * 1000, post: (t3 - t2) * 1000)
        }

        // The seed frame bundles warmup steps; its timing isn't steady-state, so don't report it.
        let timing: MatteFrame.Timing? = isSeedFrame ? nil
            : .init(preprocessMs: (t1 - t0) * 1000,
                    inferenceMs: (t2 - t1) * 1000,
                    postprocessMs: (t3 - t2) * 1000)
        completion(MatteFrame(alpha: buffer, timing: timing))
    }

    private func logStats(alpha: [Float], pre: Double, inf: Double, post: Double) {
        var mn: Float = 1, mx: Float = 0, sum: Float = 0, fg = 0
        for v in alpha { mn = min(mn, v); mx = max(mx, v); sum += v; if v > 0.5 { fg += 1 } }
        let mean = sum / Float(alpha.count)
        let cov = Float(fg) / Float(alpha.count)
        Self.log.notice("frame \(self.frameIndex): alpha min \(mn, format: .fixed(precision: 2)) max \(mx, format: .fixed(precision: 2)) mean \(mean, format: .fixed(precision: 2)) fg \(cov, format: .fixed(precision: 2)) | pre \(pre, format: .fixed(precision: 1)) inf \(inf, format: .fixed(precision: 1)) post \(post, format: .fixed(precision: 1)) ms")
        let parts = MatAnyoneCoreML.modelNames.compactMap { name -> String? in
            guard let p = MatAnyoneCoreML.profile[name], p.calls > 0 else { return nil }
            return String(format: "%@ %.1f", name.prefix(3).description, p.ms / Double(p.calls))
        }
        Self.log.notice("  models(avg ms): \(parts.joined(separator: " "))")
        MatAnyoneCoreML.profile.removeAll()   // rolling window between logs

        let memParts = ["getSimilarity", "topKSoftmax"].compactMap { name -> String? in
            guard let p = MemoryBank.profile[name], p.calls > 0 else { return nil }
            return String(format: "%@ %.1f", name.prefix(6).description, p.ms / Double(p.calls))
        }
        if !memParts.isEmpty || MatAnyoneCoreML.slowReads > 0 {
            Self.log.notice("  memory(avg ms): \(memParts.joined(separator: " ")) | slowReads \(MatAnyoneCoreML.slowReads)")
        }
        MemoryBank.profile.removeAll()
        MatAnyoneCoreML.slowReads = 0
    }
}
