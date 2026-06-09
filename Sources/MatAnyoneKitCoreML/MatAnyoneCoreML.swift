import Accelerate
import CoreML
import Foundation

/// Loads and runs the 5 MatAnyone2 Core ML programs (encoder, uncert, readout, decoder,
/// maskencoder) exported by `scripts/export.py`. Each is a stateless fixed-shape
/// `mlprogram`; the stateful glue (memory bank, affinity/readout, uncertainty blend) lives in
/// `MatAnyoneCoreMLEngine`. All tensors cross this boundary as flat row-major `[Float]` plus an
/// explicit shape, matching the PyTorch (NCHW / channel-first token) layout the models were traced
/// in.
public final class MatAnyoneCoreML {
    public struct Tensor {
        public var data: [Float]
        public var shape: [Int]
        public var count: Int { shape.reduce(1, *) }
        public init(data: [Float], shape: [Int]) { self.data = data; self.shape = shape }
    }

    public static let modelNames = ["encoder", "uncert", "readout", "decoder", "maskencoder", "objsummary"]

    private let models: [String: MLModel]
    public private(set) var modelURLs: [String: URL] = [:]
    public let workingH: Int
    public let workingW: Int

    enum LoadError: Error { case manifestMissing(String), modelMissing(String) }

    /// `modelsDir` holds the `*.mlpackage` (or compiled `*.mlmodelc`) files + `manifest.json`.
    /// `computeUnits` is the default placement; `unitOverrides` pins specific models (e.g. the
    /// ANE-hostile `readout`/`maskencoder`) to a different unit. Per-model placement lets the
    /// conv-heavy `encoder`/`decoder` run on the ANE while routing the models the A18 ANE compiler
    /// rejects to the GPU.
    public init(modelsDir: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                unitOverrides: [String: MLComputeUnits] = [:]) throws {
        let manifestURL = modelsDir.appendingPathComponent("manifest.json")
        guard let mdata = try? Data(contentsOf: manifestURL),
              let json = try? JSONSerialization.jsonObject(with: mdata) as? [String: Any]
        else { throw LoadError.manifestMissing(manifestURL.path) }

        self.workingH = json["working_h"] as? Int ?? 512
        self.workingW = json["working_w"] as? Int ?? 288

        var loaded: [String: MLModel] = [:]
        let entries = (json["models"] as? [[String: Any]]) ?? []
        for entry in entries {
            guard let name = entry["name"] as? String else { continue }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = unitOverrides[name] ?? computeUnits
            let url = try Self.resolveModelURL(name: name, dir: modelsDir)
            loaded[name] = try MLModel(contentsOf: url, configuration: cfg)
            modelURLs[name] = url
        }
        guard !loaded.isEmpty else { throw LoadError.modelMissing(modelsDir.path) }
        self.models = loaded
    }

    /// One-shot diagnostic: load the `MLComputePlan` for `name` and log every operation the ANE
    /// can't take (so we fix exactly those ops instead of guessing). Off the hot path; call once.
    @available(iOS 17.4, macOS 14.4, *)
    public func logANEUnsupportedOps(_ name: String, log: @escaping (String) -> Void) {
        guard let url = modelURLs[name] else { return }
        Task.detached(priority: .utility) {
            do {
                let cfg = MLModelConfiguration(); cfg.computeUnits = .all
                let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
                guard case let .program(program) = plan.modelStructure else { return }
                func isANE(_ d: MLComputeDevice) -> Bool { if case .neuralEngine = d { return true }; return false }
                var counts: [String: Int] = [:]
                var total = 0
                for (_, fn) in program.functions {
                    for op in fn.block.operations where op.operatorName != "const" {
                        total += 1
                        let usage = plan.deviceUsage(for: op)
                        let onANE = usage?.supported.contains(where: isANE) ?? false
                        if !onANE { counts[op.operatorName, default: 0] += 1 }
                    }
                }
                let summary = counts.sorted { $0.value > $1.value }
                    .map { "\($0.key)×\($0.value)" }.joined(separator: " ")
                let bad = counts.values.reduce(0, +)
                log("[\(name)] \(bad)/\(total) ops not ANE-eligible: \(summary.isEmpty ? "none" : summary)")
            } catch {
                log("[\(name)] compute-plan probe failed: \(error.localizedDescription)")
            }
        }
    }

    /// Prefer a prebuilt `.mlmodelc`; otherwise compile the `.mlpackage` on first use.
    private static func resolveModelURL(name: String, dir: URL) throws -> URL {
        let fm = FileManager.default
        let compiled = dir.appendingPathComponent("\(name).mlmodelc")
        if fm.fileExists(atPath: compiled.path) { return compiled }
        let pkg = dir.appendingPathComponent("\(name).mlpackage")
        guard fm.fileExists(atPath: pkg.path) else { throw LoadError.modelMissing(pkg.path) }
        return try MLModel.compileModel(at: pkg)
    }

    // ------------------------------------------------------------------- predict
    /// Per-model cumulative predict time (ms) and call count, for profiling. Off the hot path.
    public static var profile: [String: (ms: Double, calls: Int)] = [:]
    public static var profilingEnabled = false

    /// Low-level: run a model on raw `MLFeatureValue`s and return the provider. Lets the caller pass
    /// pass-through `MLMultiArray`s (e.g. encoder feature maps straight into the decoder) without
    /// any `[Float]` round-trip. Heavy maps that Swift never inspects stay on the GPU/ANE-friendly
    /// `MLMultiArray` path; only memory-touching tensors are converted via `readMultiArray`.
    public func run(_ name: String, _ feats: [String: MLFeatureValue]) throws -> MLFeatureProvider {
        guard let model = models[name] else { throw LoadError.modelMissing(name) }
        let provider = try MLDictionaryFeatureProvider(dictionary: feats)
        let t0 = Self.profilingEnabled ? CFAbsoluteTimeGetCurrent() : 0
        let out: MLFeatureProvider
        do {
            out = try model.prediction(from: provider)
        } catch {
            throw NSError(domain: "MatAnyoneCoreML", code: -6, userInfo: [
                NSLocalizedDescriptionKey: "model '\(name)' prediction failed: \(error.localizedDescription)"
            ])
        }
        if Self.profilingEnabled {
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let prev = Self.profile[name] ?? (0, 0)
            Self.profile[name] = (prev.ms + ms, prev.calls + 1)
        }
        return out
    }

    public static func value(_ t: Tensor) throws -> MLFeatureValue {
        MLFeatureValue(multiArray: try makeMultiArray(t))
    }

    // ------------------------------------------------------------------- MLMultiArray <-> [Float]
    public static func makeMultiArray(_ t: MatAnyoneCoreML.Tensor) throws -> MLMultiArray {
        guard t.data.count == t.count else {
            throw NSError(domain: "MatAnyoneCoreML", code: -10, userInfo: [
                NSLocalizedDescriptionKey:
                    "tensor data count \(t.data.count) != shape \(t.shape) product \(t.count)"
            ])
        }
        let m = try MLMultiArray(shape: t.shape.map { NSNumber(value: $0) }, dataType: .float32)
        t.data.withUnsafeBytes { src in
            memcpy(m.dataPointer, src.baseAddress!, t.data.count * MemoryLayout<Float>.stride)
        }
        return m
    }

    public static var slowReads = 0
    public static func readMultiArray(_ m: MLMultiArray) -> MatAnyoneCoreML.Tensor {
        let shape = m.shape.map { $0.intValue }
        let n = shape.reduce(1, *)
        var data = [Float](repeating: 0, count: n)

        var expected = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            expected[i] = expected[i + 1] * shape[i + 1]
        }
        let strides = m.strides.map { $0.intValue }
        let contiguous = strides == expected
        if !contiguous { slowReads += 1 }
        // Storage may exceed `n` when strides are padded; bind enough capacity for the max offset.
        var storage = 1
        for d in 0..<shape.count { storage += (shape[d] - 1) * strides[d] }
        let capacity = max(n, storage)

        // CoreML/ANE outputs are sometimes non-C-contiguous (alignment padding). Find the largest
        // *contiguous* trailing block (padding almost always lives on outer dims, e.g. the channel
        // stride, with the inner h·w plane still packed) and bulk-copy one run at a time:
        // `memcpy` for f32, SIMD `vImageConvert_Planar16FtoPlanarF` for f16. This avoids both per-
        // element scalar conversion and NSNumber boxing (~1000x slower for multi-MB tensors).
        let rank = shape.count
        var inner = 1, expStride = 1, d = rank - 1
        while d >= 0 && strides[d] == expStride { inner *= shape[d]; expStride *= shape[d]; d -= 1 }
        let outerDims = d + 1                              // dims [0, outerDims) are the strided outer
        let outerCount = inner > 0 ? n / inner : 0

        // Visit each contiguous inner run, handing its source offset + dst offset to `copyRun`.
        func forEachRun(_ copyRun: (_ srcOff: Int, _ dstOff: Int) -> Void) {
            if contiguous || outerDims == 0 { copyRun(0, 0); return }
            var idx = [Int](repeating: 0, count: outerDims)
            var dstOff = 0
            for _ in 0..<outerCount {
                var srcOff = 0
                for dim in 0..<outerDims { srcOff += idx[dim] * strides[dim] }
                copyRun(srcOff, dstOff)
                dstOff += inner
                var dim = outerDims - 1
                while dim >= 0 { idx[dim] += 1; if idx[dim] < shape[dim] { break }; idx[dim] = 0; dim -= 1 }
            }
        }
        let runLen = (contiguous || outerDims == 0) ? n : inner

        switch m.dataType {
        case .float32:
            m.dataPointer.withMemoryRebound(to: Float.self, capacity: capacity) { p in
                data.withUnsafeMutableBufferPointer { dst in
                    forEachRun { srcOff, dstOff in
                        dst.baseAddress!.advanced(by: dstOff).update(from: p + srcOff, count: runLen)
                    }
                }
            }
        case .float16:
            data.withUnsafeMutableBufferPointer { dst in
                forEachRun { srcOff, dstOff in
                    var s = vImage_Buffer(data: m.dataPointer.advanced(by: srcOff * 2),
                                          height: 1, width: vImagePixelCount(runLen), rowBytes: runLen * 2)
                    var dd = vImage_Buffer(data: UnsafeMutableRawPointer(dst.baseAddress!.advanced(by: dstOff)),
                                           height: 1, width: vImagePixelCount(runLen), rowBytes: runLen * 4)
                    vImageConvert_Planar16FtoPlanarF(&s, &dd, vImage_Flags(kvImageDoNotTile))
                }
            }
        case .double:
            m.dataPointer.withMemoryRebound(to: Double.self, capacity: capacity) { p in
                data.withUnsafeMutableBufferPointer { dst in
                    forEachRun { srcOff, dstOff in
                        for c in 0..<runLen { dst[dstOff + c] = Float(p[srcOff + c]) }
                    }
                }
            }
        @unknown default:
            for i in 0..<n { data[i] = m[i].floatValue }
        }
        return Tensor(data: data, shape: shape)
    }
}
