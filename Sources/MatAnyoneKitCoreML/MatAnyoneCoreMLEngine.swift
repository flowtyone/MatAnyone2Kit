import Accelerate
import CoreML
import Foundation

/// Stateful single-object MatAnyone2 matting on Core ML / ANE.
///
/// Drives the 5 exported Core ML programs in `MatAnyoneCoreML`, with the memory bank / affinity /
/// readout / uncertainty blend in pure Swift (`MemoryBank`, Accelerate). Mirrors the upstream
/// MatAnyone2 `InferenceCore.process_video`:
///
///     seed(frame0, seedMask)   // memorize seed, predict, warm up on the static first frame
///     step(frame_t)            // realtime frames
///
/// Spatial tensors are PyTorch-layout NCHW; memory tokens are channel-first `[C, N]`.
public final class MatAnyoneCoreMLEngine {
    public typealias Tensor = MatAnyoneCoreML.Tensor

    private let model: MatAnyoneCoreML
    private let memory: MemoryBank
    private let memEvery: Int
    private let staggerTi: Set<Int>
    private let sensoryDim: Int

    public let H: Int
    public let W: Int

    struct EngineError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
    private func need<T>(_ v: T?, _ what: String) throws -> T {
        guard let v else { throw EngineError(message: "nil \(what) (currTi=\(currTi) h=\(h) w=\(w))") }
        return v
    }
    private func out(_ p: MLFeatureProvider, _ key: String) throws -> MLMultiArray {
        guard let m = p.featureValue(for: key)?.multiArrayValue else {
            throw EngineError(message: "missing model output '\(key)'")
        }
        return m
    }

    // Feature-grid dims (stride-16), discovered from the first encode.
    private var h = 0
    private var w = 0
    private var hw: Int { h * w }

    private var currTi = -1
    private var lastMemTi = 0
    private var lastMask: [Float]?          // full-res alpha, [H*W] (Swift touches it -> [Float])
    private var lastPixFeat: MLMultiArray?  // [1,256,h,w] pass-through (never inspected in Swift)
    private var lastMskValue: [Float]?      // channel-first [CV, hw]
    private var sensory: MLMultiArray?      // [1,1,256,h,w] pass-through

    public init(model: MatAnyoneCoreML, memEvery: Int = 5, maxMemFrames: Int = 5, topK: Int = 30,
                staggerUpdates: Int = 5, sensoryDim: Int = 256) {
        self.model = model
        self.memEvery = memEvery
        self.sensoryDim = sensoryDim
        self.H = model.workingH
        self.W = model.workingW
        self.memory = MemoryBank(maxMemFrames: maxMemFrames - 1, topK: topK,
                                 keyDim: 64, valueDim: sensoryDim)

        if staggerUpdates >= memEvery {
            self.staggerTi = Set(1...memEvery)
        } else {
            var s = Set<Int>()
            let denom = Double(staggerUpdates - 1)
            for i in 0..<staggerUpdates {
                let t = 1.0 + (Double(memEvery) - 1.0) * (denom == 0 ? 0 : Double(i) / denom)
                s.insert(Int(t.rounded()))
            }
            self.staggerTi = s
        }
    }

    // ------------------------------------------------------------------- public
    /// image: [1,3,H,W] RGB in [0,1]. seedMask: [H*W] in [0,1]. Returns refined alpha [H*W].
    @discardableResult
    public func seed(image: Tensor, seedMask: [Float], warmup: Int = 10) throws -> [Float] {
        _ = try step(image, seedMask: seedMask)
        var alpha = try step(image, firstFramePred: true)!
        for _ in 0..<max(0, warmup - 1) {
            alpha = try step(image, firstFramePred: true)!
        }
        return alpha
    }

    /// image: [1,3,H,W] RGB in [0,1]. Returns alpha [H*W] in [0,1].
    @discardableResult
    public func step(image: Tensor) throws -> [Float] { try step(image, seedMask: nil)! }

    // ------------------------------------------------------------------- step
    @discardableResult
    func step(_ image: Tensor, seedMask: [Float]? = nil,
              firstFramePred: Bool = false) throws -> [Float]? {
        currTi += 1
        var isMemFrame = (currTi - lastMemTi >= memEvery) || (seedMask != nil)
        var needSegment = seedMask == nil
        var updateSensory = staggerTi.contains(currTi - lastMemTi)
        if firstFramePred {
            currTi = 0; lastMemTi = 0
            isMemFrame = true; needSegment = true; updateSensory = true
        }

        // --- encode (feature maps f16..f1 + pix_feat stay as MLMultiArray pass-through)
        let enc = try model.run("encoder", ["image": try MatAnyoneCoreML.value(image)])
        let pixFeat = try mv(enc, "pix_feat")
        let key = try tensor(enc, "key"), shrinkage = try tensor(enc, "shrinkage")
        let selection = try tensor(enc, "selection")
        if h == 0 { h = pixFeat.shape[2].intValue; w = pixFeat.shape[3].intValue; initSensoryIfNeeded() }

        // --- segment
        var alpha: [Float]?
        if needSegment {
            alpha = try segment(enc: enc, pixFeat: pixFeat, key: key, selection: selection,
                                updateSensory: updateSensory)
        }
        if let seedMask { alpha = seedMask }

        lastMask = alpha
        lastPixFeat = pixFeat

        // --- memory write (mask_value identical regardless of deep_update; commit only on mem frames)
        if firstFramePred { memory.clearTemp() }
        let masksFV = try fv(try need(lastMask, "lastMask"), [1, 1, H, W])
        let me = try model.run("maskencoder", [
            "image": try MatAnyoneCoreML.value(image),
            "pix_feat": MLFeatureValue(multiArray: pixFeat),
            "sensory": MLFeatureValue(multiArray: try need(sensory, "sensory")),
            "masks": masksFV,
        ])
        let maskValue = try mv(me, "mask_value")               // [1,CV,h,w] pass-through
        lastMskValue = MatAnyoneCoreML.readMultiArray(maskValue).data
        if isMemFrame {
            // object_summarizer split out (ANE-hostile) — only needed when we commit a memory frame.
            let os = try model.run("objsummary", [
                "masks": masksFV,
                "mask_value": MLFeatureValue(multiArray: maskValue),
            ])
            let objSum = try tensor(os, "obj_summaries")
            memory.addMemory(key: key.data, shrinkage: shrinkage.data,
                             value: try need(lastMskValue, "lastMskValue"),
                             objValue: objSum.data, objValueShape: objSum.shape,
                             h: h, w: w, asPermanent: true)
            lastMemTi = currTi
            sensory = try mv(me, "new_sensory")                // deep update
        }
        return alpha
    }

    // ------------------------------------------------------------------- segment
    private func segment(enc: MLFeatureProvider, pixFeat: MLMultiArray, key: Tensor,
                         selection: Tensor, updateSensory: Bool) throws -> [Float] {
        let visual: [Float]                                    // channel-first [CV, hw]
        let lastMsk = try need(lastMskValue, "lastMskValue")
        if currTi == 0 {
            visual = lastMsk                                   // first frame: no uncertainty blend
        } else {
            let affinity = memory.matchMemory(queryKey: key.data, querySelection: selection.data)
            let readout = memory.readoutValue(affinity)        // [CV, hw]
            var diff = [Float](repeating: 0, count: readout.count)
            vDSP.subtract(readout, lastMsk, result: &diff)
            let prob = try tensor(try model.run("uncert", [
                "last_pix_feat": MLFeatureValue(multiArray: try need(lastPixFeat, "lastPixFeat")),
                "cur_pix_feat": MLFeatureValue(multiArray: pixFeat),
                "last_mask": try fv(try need(lastMask, "lastMask"), [1, 1, H, W]),
                "mem_val_diff": try fv(diff, [1, sensoryDim, h, w]),
            ]), "prob")                                        // [1,1,h,w] -> per-pixel
            visual = blend(readout: readout, last: lastMsk, prob: prob.data)
        }

        let ro = try model.run("readout", [
            "pix_feat": MLFeatureValue(multiArray: pixFeat),
            "pixel": try fv(visual, [1, 1, sensoryDim, h, w]),
            "sensory": MLFeatureValue(multiArray: try need(sensory, "sensory")),
            "last_mask": try fv(try need(lastMask, "lastMask"), [1, 1, H, W]),
            "obj_memory": try fv(try need(memory.objV, "memory.objV"), [1, 1, 1, 16, sensoryDim + 1]),
        ])

        let dec = try model.run("decoder", [
            "f16": try mfv(enc, "f16"), "f8": try mfv(enc, "f8"), "f4": try mfv(enc, "f4"),
            "f2": try mfv(enc, "f2"), "f1": try mfv(enc, "f1"),
            "memory_readout": try mfv(ro, "mem_readout"),
            "sensory": MLFeatureValue(multiArray: try need(sensory, "sensory")),
        ])
        if updateSensory { sensory = try mv(dec, "new_sensory") }

        var alpha = (try tensor(dec, "logits")).data           // [1,1,H,W]
        clampInPlace(&alpha, 0, 1)
        return alpha
    }

    // ------------------------------------------------------------------- helpers
    /// Convert a memory-touching output to `[Float]`.
    private func tensor(_ p: MLFeatureProvider, _ key: String) throws -> Tensor {
        MatAnyoneCoreML.readMultiArray(try out(p, key))
    }
    /// Grab an output as a raw `MLMultiArray` (pass-through, no conversion).
    private func mv(_ p: MLFeatureProvider, _ key: String) throws -> MLMultiArray {
        try out(p, key)
    }
    /// Re-feed an output straight back in as an input feature value.
    private func mfv(_ p: MLFeatureProvider, _ key: String) throws -> MLFeatureValue {
        MLFeatureValue(multiArray: try out(p, key))
    }
    /// Wrap a Swift `[Float]` + shape as an input feature value.
    private func fv(_ data: [Float], _ shape: [Int]) throws -> MLFeatureValue {
        try MatAnyoneCoreML.value(Tensor(data: data, shape: shape))
    }

    private func initSensoryIfNeeded() {
        if sensory == nil {
            sensory = try? MatAnyoneCoreML.makeMultiArray(
                Tensor(data: [Float](repeating: 0, count: sensoryDim * hw),
                       shape: [1, 1, sensoryDim, h, w]))
        }
    }

    /// `visual[c,j] = readout[c,j]*prob[j] + last[c,j]*(1-prob[j])` (prob broadcast over channels),
    /// computed as `last + prob·(readout - last)` with a vDSP subtract + multiply-add per channel.
    private func blend(readout: [Float], last: [Float], prob: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: readout.count)
        let n = vDSP_Length(hw)
        readout.withUnsafeBufferPointer { ro in
        last.withUnsafeBufferPointer { la in
        prob.withUnsafeBufferPointer { pr in
        out.withUnsafeMutableBufferPointer { ot in
            let p = pr.baseAddress!
            for c in 0..<sensoryDim {
                let base = c * hw
                let roc = ro.baseAddress! + base
                let lac = la.baseAddress! + base
                let otc = ot.baseAddress! + base
                vDSP_vsub(lac, 1, roc, 1, otc, 1, n)        // otc = readout - last
                vDSP_vma(otc, 1, p, 1, lac, 1, otc, 1, n)   // otc = (readout-last)*prob + last
            }
        }}}}
        return out
    }

    private func clampInPlace(_ a: inout [Float], _ lo: Float, _ hi: Float) {
        var l = lo, hr = hi
        vDSP_vclip(a, 1, &l, &hr, &a, 1, vDSP_Length(a.count))
    }
}
