// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MatAnyoneKitCoreML",
    platforms: [
        .iOS(.v17),     // MLComputePlan ANE probe is iOS 17.4+ (guarded internally)
        .macOS(.v14),
    ],
    products: [
        .library(name: "MatAnyoneKitCoreML", targets: ["MatAnyoneKitCoreML"]),
    ],
    targets: [
        .target(
            name: "MatAnyoneKitCoreML",
            path: "Sources/MatAnyoneKitCoreML",
            // The 6 precompiled MatAnyone2 models (.mlmodelc) + manifest.json ship inside the
            // package, so a consumer just adds the dependency and gets matting — no separate
            // model download or build-time compile step.
            resources: [.copy("Resources/MatAnyone")]
        ),
    ]
)
