import Accelerate
import Foundation

/// Single-object MatAnyone working memory + affinity/readout, in pure Swift/Accelerate.
///
/// Implements the upstream MatAnyone2 `SingleObjectMemory` / `MemoryOps` in Swift. Batch is
/// always 1 in the realtime path (no flip-aug), so everything is 2-D row-major `[Float]` and the
/// matmuls are `cblas_sgemm`. Token layout is channel-first `[C, N]`, matching the PyTorch memory
/// boundary; `N = T * h * w` grows each memory frame and is FIFO-capped.
///
/// Validated end-to-end against the PyTorch reference by `scripts/dump_e2e_ref.py` + `e2e_validate.swift`.
final class MemoryBank {
    let maxMemFrames: Int
    let topK: Int
    let keyDim: Int      // CK
    let valueDim: Int    // CV

    private(set) var h = 0
    private(set) var w = 0

    // Channel-first stores, row-major: key[CK,N], shrinkage[N], value[CV,N].
    private var key: [Float] = []
    private var shrinkage: [Float] = []
    private var value: [Float] = []
    private var n = 0
    private var permEnd = 0

    // Object memory running sum [Q, embedDim+1] and sensory [h, w, sensoryDim] (NHWC).
    private(set) var objV: [Float]?
    private(set) var objVShape: [Int]?

    init(maxMemFrames: Int = 4, topK: Int = 30, keyDim: Int = 64, valueDim: Int = 256) {
        self.maxMemFrames = maxMemFrames
        self.topK = topK
        self.keyDim = keyDim
        self.valueDim = valueDim
    }

    var hw: Int { h * w }
    var maxWorkTokens: Int { maxMemFrames * hw }
    var engaged: Bool { n > 0 }

    func clearTemp() {
        key = []; shrinkage = []; value = []
        n = 0; permEnd = 0
        objV = nil; objVShape = nil
    }

    // ----------------------------------------------------------------- write
    /// keyIn[CK,h,w], shrinkageIn[1,h,w], mskValue[CV,h,w] (channel-first, contiguous),
    /// objValue[Q, C+1]. Appends a memory frame and FIFO-caps the working tokens.
    func addMemory(key keyIn: [Float], shrinkage shrinkageIn: [Float], value mskValue: [Float],
                   objValue: [Float], objValueShape: [Int], h: Int, w: Int, asPermanent: Bool) {
        self.h = h; self.w = w
        let newN = h * w

        accumulateObj(objValue, objValueShape)
        appendTokens(into: &key, src: keyIn, rows: keyDim, addCols: newN)
        appendTokens(into: &shrinkage, src: shrinkageIn, rows: 1, addCols: newN)
        appendTokens(into: &value, src: mskValue, rows: valueDim, addCols: newN)
        n += newN
        if asPermanent && permEnd == 0 { permEnd = n }
        fifo()
    }

    private func accumulateObj(_ v: [Float], _ shape: [Int]) {
        if objV == nil {
            objV = v; objVShape = shape
        } else {
            vDSP.add(objV!, v, result: &objV!)
        }
    }

    /// Append `addCols` columns to a row-major `[rows, oldCols]` store → `[rows, oldCols+addCols]`.
    private func appendTokens(into store: inout [Float], src: [Float], rows: Int, addCols: Int) {
        let oldCols = rows == 0 ? 0 : (store.count / rows)
        let newCols = oldCols + addCols
        var out = [Float](repeating: 0, count: rows * newCols)
        for r in 0..<rows {
            for c in 0..<oldCols { out[r * newCols + c] = store[r * oldCols + c] }
            for c in 0..<addCols { out[r * newCols + oldCols + c] = src[r * addCols + c] }
        }
        store = out
    }

    /// Keep the permanent prefix + the most recent `maxWorkTokens` temporary tokens.
    private func fifo() {
        let nonPerm = n - permEnd
        if nonPerm <= maxWorkTokens { return }
        let keepStart = n - maxWorkTokens
        let keptCols = permEnd + (n - keepStart)
        func sieve(_ store: [Float], rows: Int) -> [Float] {
            var out = [Float](repeating: 0, count: rows * keptCols)
            for r in 0..<rows {
                var dst = r * keptCols
                for c in 0..<permEnd { out[dst] = store[r * n + c]; dst += 1 }
                for c in keepStart..<n { out[dst] = store[r * n + c]; dst += 1 }
            }
            return out
        }
        key = sieve(key, rows: keyDim)
        shrinkage = sieve(shrinkage, rows: 1)
        value = sieve(value, rows: valueDim)
        n = keptCols
    }

    // ----------------------------------------------------------------- read
    static var profile: [String: (ms: Double, calls: Int)] = [:]
    static var profilingEnabled = false
    @inline(__always) private func lap(_ name: String, _ t0: CFAbsoluteTime) {
        guard Self.profilingEnabled else { return }
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let p = Self.profile[name] ?? (0, 0)
        Self.profile[name] = (p.ms + ms, p.calls + 1)
    }

    /// qk,qe: [CK, h, w] channel-first. Returns affinity [N, HW] (row-major).
    func matchMemory(queryKey qk: [Float], querySelection qe: [Float]) -> [Float] {
        let t0 = CFAbsoluteTimeGetCurrent()
        let sim = MemoryMath.getSimilarity(key: key, shrinkage: shrinkage, qk: qk, qe: qe,
                                           ck: keyDim, n: n, hw: hw)
        lap("getSimilarity", t0)
        let t1 = CFAbsoluteTimeGetCurrent()
        let r = MemoryMath.topKSoftmax(sim, rows: n, cols: hw, k: min(topK, n))
        lap("topKSoftmax", t1)
        return r
    }

    /// affinity [N, HW] → readout [CV, h, w] (channel-first, contiguous).
    func readoutValue(_ affinity: [Float]) -> [Float] {
        MemoryMath.readout(value: value, affinity: affinity, cv: valueDim, n: n, hw: hw)
    }
}

/// Pure, stateless memory math (anisotropic-L2 similarity, top-k softmax, readout). Extracted so
/// it can be exercised on the Mac via the `scripts/` PyTorch parity harness independent of Core ML.
/// All arrays are row-major `[Float]`; batch is 1.
enum MemoryMath {
    /// mk=key[CK,N], ms=shrinkage[N], qk/qe=[CK,HW] → sim[N,HW].
    static func getSimilarity(key mk: [Float], shrinkage ms: [Float], qk: [Float], qe: [Float],
                              ck: Int, n: Int, hw: Int) -> [Float] {
        var qkqe = [Float](repeating: 0, count: ck * hw)
        vDSP.multiply(qk, qe, result: &qkqe)               // qk⊙qe  [CK,HW]
        var mk2 = [Float](repeating: 0, count: ck * n)
        vDSP.multiply(mk, mk, result: &mk2)                // mk⊙mk  [CK,N]

        var aSq = [Float](repeating: 0, count: n * hw)     // (mk²)^T @ qe
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Int32(n), Int32(hw), Int32(ck),
                    1.0, mk2, Int32(n), qe, Int32(hw), 0.0, &aSq, Int32(hw))
        var twoAb = [Float](repeating: 0, count: n * hw)   // 2 · mk^T @ (qk⊙qe)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Int32(n), Int32(hw), Int32(ck),
                    2.0, mk, Int32(n), qkqe, Int32(hw), 0.0, &twoAb, Int32(hw))

        var qkqkqe = [Float](repeating: 0, count: ck * hw)
        vDSP.multiply(qkqe, qk, result: &qkqkqe)           // qe⊙qk⊙qk
        var bSq = [Float](repeating: 0, count: hw)         // sum_ck
        qkqkqe.withUnsafeBufferPointer { q in
        bSq.withUnsafeMutableBufferPointer { b in
            for c in 0..<ck {
                let base = c * hw
                for j in 0..<hw { b[j] += q[base + j] }
            }
        }}

        let invSqrtCk = 1.0 / Float(ck).squareRoot()
        var sim = [Float](repeating: 0, count: n * hw)
        sim.withUnsafeMutableBufferPointer { s in
        aSq.withUnsafeBufferPointer { a in
        twoAb.withUnsafeBufferPointer { t in
        bSq.withUnsafeBufferPointer { b in
        ms.withUnsafeBufferPointer { msP in
            for i in 0..<n {
                let row = i * hw
                let msScale = msP[i] * invSqrtCk
                for j in 0..<hw { s[row + j] = (-a[row + j] + t[row + j] - b[j]) * msScale }
            }
        }}}}}
        return sim
    }

    /// Top-k softmax over rows (memory dim) per column. Non-top entries → 0.
    ///
    /// Columns are independent, so the work is split across cores with `concurrentPerform`. Each
    /// worker owns private scratch and writes only its own column range of `out` (disjoint indices),
    /// so no synchronization is needed. Math is identical to the serial version.
    static func topKSoftmax(_ sim: [Float], rows: Int, cols: Int, k: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        let workers = max(1, min(cols, ProcessInfo.processInfo.activeProcessorCount))
        let chunk = (cols + workers - 1) / workers
        sim.withUnsafeBufferPointer { simP in
        out.withUnsafeMutableBufferPointer { outP in
            let simBase = simP.baseAddress!
            let outBase = outP.baseAddress!
            DispatchQueue.concurrentPerform(iterations: workers) { t in
                let cStart = t * chunk
                guard cStart < cols else { return }
                let cEnd = min(cStart + chunk, cols)
                var col = [Float](repeating: 0, count: rows)
                var sel = [Float](repeating: 0, count: rows)
                var survIdx = [Int](repeating: 0, count: rows)   // top-k survivors (≈k, more on ties)
                var survVal = [Float](repeating: 0, count: rows)
                col.withUnsafeMutableBufferPointer { colP in
                sel.withUnsafeMutableBufferPointer { selP in
                survIdx.withUnsafeMutableBufferPointer { sIdx in
                survVal.withUnsafeMutableBufferPointer { sVal in
                    for j in cStart..<cEnd {
                        for i in 0..<rows { let v = simBase[i * cols + j]; colP[i] = v; selP[i] = v }
                        let thr = kthLargest(selP, k: k)           // O(N) quickselect
                        var maxV: Float = 0
                        vDSP_maxv(colP.baseAddress!, 1, &maxV, vDSP_Length(rows))   // SIMD column max
                        var cnt = 0                                 // collect survivors in one pass
                        for i in 0..<rows where colP[i] >= thr { sIdx[cnt] = i; sVal[cnt] = colP[i]; cnt += 1 }
                        var sum: Float = 0
                        for s in 0..<cnt { let e = expf(sVal[s] - maxV); sVal[s] = e; sum += e }
                        if sum > 0 {
                            let inv = 1.0 / sum
                            for s in 0..<cnt { outBase[sIdx[s] * cols + j] = sVal[s] * inv }
                        }
                    }
                }}}}
            }
        }}
        return out
    }

    /// value[CV,N] @ affinity[N,HW] → [CV,HW].
    static func readout(value: [Float], affinity: [Float], cv: Int, n: Int, hw: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cv * hw)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(cv), Int32(hw), Int32(n),
                    1.0, value, Int32(n), affinity, Int32(hw), 0.0, &out, Int32(hw))
        return out
    }

    /// kth-largest value via in-place quickselect (Hoare). Mutates `a`.
    static func kthLargest(_ a: UnsafeMutableBufferPointer<Float>, k: Int) -> Float {
        var lo = 0, hi = a.count - 1
        let target = k - 1                                  // 0-based index of kth largest (desc)
        while lo < hi {
            let pivot = a[(lo + hi) / 2]
            var i = lo, j = hi
            while i <= j {
                while a[i] > pivot { i += 1 }               // descending order
                while a[j] < pivot { j -= 1 }
                if i <= j { a.swapAt(i, j); i += 1; j -= 1 }
            }
            if target <= j { hi = j } else if target >= i { lo = i } else { break }
        }
        return a[target]
    }
}
