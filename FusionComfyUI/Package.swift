// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FusionComfyUI",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "FusionComfyUI",
            path: ".",
            exclude: ["Scripts"]
        ),
    ]
)
