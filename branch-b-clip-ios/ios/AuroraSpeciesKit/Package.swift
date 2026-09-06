// swift-tools-version:5.9
import PackageDescription

// AuroraSpeciesKit — native on-device species classification for iPhone (A15+).
//
// Resources expected in Sources/AuroraSpeciesKit/Resources (see ../README.md):
//   BioCLIP2-ImageEncoder.mlmodelc   (compiled from the mlpackage with
//                                     `xcrun coremlcompiler compile` on a Mac)
//   species_embeddings.f16.bin       (504x768 fp16, 0.77 MB)
//   species_table.json               (names/sci/group/danger + format metadata)
let package = Package(
    name: "AuroraSpeciesKit",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [.library(name: "AuroraSpeciesKit", targets: ["AuroraSpeciesKit"])],
    targets: [
        .target(
            name: "AuroraSpeciesKit",
            resources: [
                .copy("Resources/species_table.json"),
                .copy("Resources/species_embeddings.f16.bin"),
                .copy("Resources/BioCLIP2-ImageEncoder.mlmodelc"),
            ]
        ),
        .testTarget(
            name: "AuroraSpeciesKitTests",
            dependencies: ["AuroraSpeciesKit"],
            resources: [.copy("Photos")]  // drop the exam photos/ folder + photo_labels.json here
        ),
    ]
)
