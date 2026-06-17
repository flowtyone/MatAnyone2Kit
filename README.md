# MatAnyone2Kit

[![code: GPL-3.0](https://img.shields.io/badge/code-GPL--3.0-blue)](LICENSE)
[![weights: S-Lab 1.0 non-commercial](https://img.shields.io/badge/weights-S--Lab%201.0%20non--commercial-red)](Sources/MatAnyoneKitCoreML/Resources/MatAnyone/LICENSE)

A self-contained Swift package that runs the [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
single-object video-matting model **in real time on the Apple Neural Engine** — stable 30 fps on an
iPhone 16 (A18). The six precompiled Core ML models ship inside the package, so you add one
dependency and feed it camera frames; there's no separate model download or launch-time compile.

<p align="center">
  <img src="scripts/example/compare.gif" width="420" alt="Left: raw camera frame. Right: the same frame matted by MatAnyone2 on the Apple Neural Engine.">
</p>

<p align="center"><em>Same frame, split down the middle — raw camera on the left, MatAnyone2's real-time matte on the right. No green screen, no rotoscoping.</em></p>

The conversion toolchain (and a writeup of every ANE/Swift optimization that made it real-time) lives
in [`scripts/`](scripts).

> **License:** the package code is GPL-3.0, but the bundled MatAnyone2 weights are
> [NTU S-Lab License 1.0 — **non-commercial only**](#license). Using this package with the bundled
> weights is non-commercial. [Details ↓](#license)

## Install

Swift Package Manager — add it as a dependency in your `Package.swift`:

```swift
.package(url: "https://github.com/flowtyone/MatAnyone2Kit", from: "1.0.0")
// then add "MatAnyoneKitCoreML" to your target's dependencies
```

Or in Xcode: **File ▸ Add Package Dependencies…** and paste
`https://github.com/flowtyone/MatAnyone2Kit`.

Requires iOS 17+ / macOS 14+ (the on-device ANE-eligibility probe uses `MLComputePlan`, iOS 17.4 /
macOS 14.4, guarded internally).

## Use

```swift
import MatAnyoneKitCoreML

// Loads the bundled, precompiled models; nil if they can't be loaded.
guard let matte = MatAnyoneMatte() else { return }

// On each camera frame (BGRA CVPixelBuffer). The first frame with a person seeds the
// tracker via Vision; subsequent frames track from MatAnyone's own memory.
matte.matte(pixelBuffer) { frame in
    guard let alpha = frame.alpha else {
        // .passthrough — no matte yet (no person seeded). Show the full camera frame.
        return
    }
    // `alpha` is a single-channel CVPixelBuffer (OneComponent8, 1 = foreground) at the model
    // working resolution (288×512), framed like the camera frame. Composite it however you like.
    if let timing = frame.timing {
        // per-stage wall-clock (ms): timing.preprocessMs / inferenceMs / postprocessMs
    }
}
```

`matte(_:completion:)` runs synchronously on the calling queue (typically your camera/video output
queue) and calls `completion` once. For best throughput, drive it back-to-back keeping only the
freshest pending frame rather than once per camera tick.

## Real-time best practices

These are the settings that get a stable **30 fps on an iPhone 16 (A18)** in a live camera app.

**Resolution.** The bundled models run at a fixed **288×512** (portrait) — that's the working size
tuned for 30 fps on the ANE, and you don't choose it (`matte.workingWidth/workingHeight` report it).
So you control cost *upstream*, at capture: feed **720p or lower**. Capturing at 4K just burns ISP
and conversion time for frames the matte immediately downsizes to 288×512. Capture BGRA
(`kCVPixelFormatType_32BGRA`), upright/portrait — the returned alpha is framed identically to the
frame you pass in, so you composite it with the same aspect-fill UVs as the camera.

```swift
session.sessionPreset = .hd1280x720   // 720p is plenty; the matte works at 288×512
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
output.alwaysDiscardsLateVideoFrames = true
```

**Capture at 60 fps, not 30.** The matte's throughput quantum is the camera tick. At 30 fps (33 ms
ticks) a 34 ms matte misses *every* tick and snaps to 15 fps. Pinning the camera to 60 fps (16.7 ms
ticks) lets a sub-33 ms matte hold 30 fps, with a 20 fps floor otherwise. Pinning `min == max` also
stops iOS from auto-dropping the rate in low light.

```swift
if let maxRate = device.activeFormat.videoSupportedFrameRateRanges.map(\.maxFrameRate).max(),
   (try? device.lockForConfiguration()) != nil {
    let dur = CMTime(value: 1, timescale: CMTimeScale(min(60.0, maxRate).rounded()))
    device.activeVideoMinFrameDuration = dur
    device.activeVideoMaxFrameDuration = dur
    device.unlockForConfiguration()
}
```

**Drive it back-to-back, don't drop on the camera tick.** Run the matte off the main thread on one
serial queue, and when a frame arrives mid-pass keep only the *freshest* one, then process it the
instant the current pass finishes. This decouples throughput from the 33 ms delivery window (a 35 ms
matte streams at ~28 fps instead of collapsing to 15) while keeping latency at ~one frame:

```swift
final class MattePacer {
    private let matte: MatAnyoneMatte
    private let queue = DispatchQueue(label: "matte", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var pending: CVPixelBuffer?
    var onResult: ((CVPixelBuffer, MatAnyoneMatte.MatteFrame) -> Void)?

    init(_ matte: MatAnyoneMatte) { self.matte = matte }

    // Call from your camera delegate on every frame.
    func submit(_ frame: CVPixelBuffer) {
        lock.lock()
        if busy { pending = frame; lock.unlock(); return }   // keep only the freshest
        busy = true; lock.unlock()
        process(frame)
    }

    private func process(_ frame: CVPixelBuffer) {
        queue.async { [self] in
            matte.matte(frame) { result in
                onResult?(frame, result)                     // source frame + its matte, in sync
                lock.lock(); let next = pending; pending = nil
                if next == nil { busy = false }; lock.unlock()
                if let next { process(next) }                // run now, not on the next camera tick
            }
        }
    }
}
```

**Composite the frame with *its own* matte.** `onResult` hands you the exact source frame each matte
was computed from — composite that pair so the cutout never lags the camera by a frame.

**Load off the main thread.** First-launch ANE specialization takes a few seconds; construct
`MatAnyoneMatte()` on a background task and run a passthrough (full-frame camera) until it's ready,
so your preview appears instantly.

```swift
Task.detached(priority: .utility) { let matte = MatAnyoneMatte(); /* swap it in when ready */ }
```

**Mind the thermal envelope.** The matte runs on the ANE, leaving the GPU free — but a heavy GPU
renderer *plus* the matte *plus* the camera ISP sustained together will throttle the device before
`ProcessInfo.thermalState` even reports `fair`. Watch per-stage latency, not just `thermalState`.

## What's inside

| type                     | role                                                                       |
|--------------------------|----------------------------------------------------------------------------|
| `MatAnyoneMatte`         | top-level facade: Vision seeding + stateful tracking, returns `MatteFrame` |
| `MatAnyoneCoreMLEngine`  | stateful inference loop over the six models                                |
| `MatAnyoneCoreML`        | model loading + efficient `MLMultiArray` ↔ `[Float]` conversion            |
| `MemoryBank` / `MemoryMath` | key/affinity memory, top-k softmax readout (Accelerate, parallelized)  | 

The lower-level engine (`MatAnyoneCoreMLEngine`, `MatAnyoneCoreML`) is public if you want to drive the
pipeline directly; most callers only need `MatAnyoneMatte`.

## Configuration

```swift
MatAnyoneMatte.diagnostics = true            // verbose per-frame logging + ANE-eligibility probe
MatAnyoneMatte.defaultUnit  = .cpuAndNeuralEngine
MatAnyoneMatte.unitOverrides = ["objsummary": .cpuAndGPU]   // per-model compute placement

// Load models from a custom directory instead of the bundled ones:
let matte = MatAnyoneMatte(modelsDir: someURL)
```

Set these **before** constructing `MatAnyoneMatte`.

## License

MatAnyone2Kit ships under **two** licenses — see [`NOTICE.md`](NOTICE.md) for the full breakdown:

- **Swift package source code** — GNU GPL-3.0 ([`LICENSE`](LICENSE)).
- **Bundled MatAnyone2 model weights** (the Core ML models in
  [`Sources/MatAnyoneKitCoreML/Resources/MatAnyone/`](Sources/MatAnyoneKitCoreML/Resources/MatAnyone/))
  — **NTU S-Lab License 1.0, non-commercial only**. These are a Core ML conversion of the
  [MatAnyone2](https://github.com/pq-yang/MatAnyone2) weights by S-Lab, NTU; converting them does not
  change their license.

> ⚠️ **Using this package with the bundled weights is non-commercial only.** The GPL-3.0 on the code
> does not grant any commercial rights to the weights. For commercial use of the weights, contact the
> authors (see the weights
> [`NOTICE.md`](Sources/MatAnyoneKitCoreML/Resources/MatAnyone/NOTICE.md)).
