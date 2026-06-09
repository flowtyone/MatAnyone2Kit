# MatAnyoneKitCoreML

A self-contained Swift package that runs the [MatAnyone2](https://github.com/pq-yang/MatAnyone)
single-object video-matting model **in real time on the Apple Neural Engine** — stable 30 fps on an
iPhone 16 (A18). The six precompiled Core ML models ship inside the package, so you add one
dependency and feed it camera frames; there's no separate model download or launch-time compile.

The conversion toolchain (and a writeup of every ANE/Swift optimization that made it real-time) lives
in [`scripts/`](scripts).

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
