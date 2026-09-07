// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AuroraSpeciesKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "AuroraSpeciesKit", targets: ["AuroraSpeciesKit"]),
    ],
    targets: [
        .target(
            name: "AuroraSpeciesKit",
            path: "species-classifier/ios/AuroraSpeciesKit/Sources/AuroraSpeciesKit"
        ),
        .testTarget(
            name: "AuroraSpeciesKitTests",
            dependencies: ["AuroraSpeciesKit"],
            path: "species-classifier/ios/AuroraSpeciesKit/Tests/AuroraSpeciesKitTests"
        ),
    ]
)
