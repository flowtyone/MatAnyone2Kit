// swift-tools-version: 5.9
import PackageDescription

// Standalone offline runner for the kit: decode an mp4, matte every frame, re-encode.
// Lives outside the library package so the shipped product stays clean; it depends on
// MatAnyoneKitCoreML by path.
let package = Package(
    name: "matte-demo",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "MatteVideo",
            dependencies: [
                .product(name: "MatAnyoneKitCoreML", package: "MatAnyone2Kit"),
            ],
            path: "Sources/MatteVideo"
        ),
    ]
)
