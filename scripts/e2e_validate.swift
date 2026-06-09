// End-to-end parity harness for the Swift Core ML matting pipeline.
//
// Run from scripts/ (sources live in the MatAnyoneKitCoreML package):
//   PYTHONPATH=/tmp/matanyone2:. uv run python dump_e2e_ref.py
//   swiftc -O -parse-as-library e2e_validate.swift \
//     ../Sources/MatAnyoneKitCoreML/MemoryBank.swift \
//     ../Sources/MatAnyoneKitCoreML/MatAnyoneCoreML.swift \
//     ../Sources/MatAnyoneKitCoreML/MatAnyoneCoreMLEngine.swift \
//     -o /tmp/e2evalidate && /tmp/e2evalidate
//
// Replays the exact reference step pattern (dump_e2e_ref.py) through the Swift engine and the
// exported .mlpackage models, then compares alphas against the PyTorch reference dump.

import CoreML
import Foundation

struct E2EManifest: Decodable { let H, W, T, n_warmup, n_out: Int }

@main
enum E2EValidate {
    static let refDir = "/tmp/e2eref"
    static let modelsDir = FileManager.default.currentDirectoryPath + "/coreml/models"

    static func loadBin(_ path: String) -> [Float] {
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func main() throws {
        let man = try JSONDecoder().decode(
            E2EManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: "\(refDir)/manifest.json")))
        let H = man.H, W = man.W, T = man.T, nWarmup = man.n_warmup
        let frameSize = 3 * H * W
        let alphaSize = H * W

        let frames = loadBin("\(refDir)/frames.bin")
        let seed = loadBin("\(refDir)/seed.bin")              // [H*W] in [0,255]
        let refAlphas = loadBin("\(refDir)/alphas.bin")       // [n_out, H*W]
        let seed01 = seed.map { $0 / 255.0 }                   // engine wants [0,1]

        func frame(_ ti: Int) -> MatAnyoneCoreML.Tensor {
            let lo = ti * frameSize
            return .init(data: Array(frames[lo..<lo + frameSize]), shape: [1, 3, H, W])
        }

        print("loading Core ML models from \(modelsDir) ...")
        let model = try MatAnyoneCoreML(modelsDir: URL(fileURLWithPath: modelsDir))
        MatAnyoneCoreML.profilingEnabled = true
        MemoryBank.profilingEnabled = true
        let engine = MatAnyoneCoreMLEngine(model: model)

        var alphas: [[Float]] = []
        let t0 = Date()
        var stepTimes: [Double] = []
        for ti in 0..<T {
            let img = frame(ti)
            var a: [Float]?
            let s = Date()
            if ti == 0 {
                _ = try engine.step(img, seedMask: seed01)
                a = try engine.step(img, firstFramePred: true)
            } else if ti <= nWarmup {
                a = try engine.step(img, firstFramePred: true)
            } else {
                a = try engine.step(img)
            }
            stepTimes.append(Date().timeIntervalSince(s) * 1000)
            if ti >= nWarmup { alphas.append(a!) }
        }
        print(String(format: "ran %d steps in %.2fs", T, Date().timeIntervalSince(t0)))

        precondition(alphas.count * alphaSize == refAlphas.count,
                     "alpha count mismatch \(alphas.count) vs \(refAlphas.count / alphaSize)")

        var allMax: Float = 0
        var allSum: Double = 0
        var allCount = 0
        var over10 = 0
        for (i, a) in alphas.enumerated() {
            var maxD: Float = 0, sum: Double = 0, o10 = 0
            for j in 0..<alphaSize {
                let d = abs(a[j] - refAlphas[i * alphaSize + j])
                if d > maxD { maxD = d }
                sum += Double(d)
                if d > 0.1 { o10 += 1 }
            }
            allMax = max(allMax, maxD); allSum += sum; allCount += alphaSize; over10 += o10
            print(String(format: "  frame %d: maxΔ %.3f  meanΔ %.5f  >.1 %.3f%%",
                         i, maxD, sum / Double(alphaSize), 100 * Double(o10) / Double(alphaSize)))
        }
        let mean = allSum / Double(allCount)
        let medStep = stepTimes.sorted()[stepTimes.count / 2]
        print(String(format: "\noverall: meanΔ %.5f  maxΔ %.3f  >.1 %.3f%%  | median step %.1f ms",
                     mean, allMax, 100 * Double(over10) / Double(allCount), medStep))

        print("per-model predict (avg ms over calls):")
        for name in ["encoder", "uncert", "readout", "decoder", "maskencoder"] {
            if let p = MatAnyoneCoreML.profile[name] {
                print(String(format: "  %-12s %6.1f ms  x%d", (name as NSString).utf8String!,
                             p.ms / Double(p.calls), p.calls))
            }
        }

        for name in ["getSimilarity", "topKSoftmax"] {
            if let p = MemoryBank.profile[name] {
                print(String(format: "  %-12s %6.1f ms  x%d", (name as NSString).utf8String!,
                             p.ms / Double(p.calls), p.calls))
            }
        }
        print("slow (NSNumber) array reads: \(MatAnyoneCoreML.slowReads)")

        // Edge-only fp16 noise (matches the Python CoreML harness: meanΔ ~0.0014, maxΔ ~0.66).
        if mean < 0.01 && 100 * Double(over10) / Double(allCount) < 1.0 {
            print("PASS")
        } else {
            print("FAIL")
            exit(1)
        }
    }
}
