import Accelerate
import CoreML
import Foundation
import ImageIO

public struct SpeciesArtifactSet: Sendable {
    public let encoderURL: URL
    public let speciesTableURL: URL
    public let embeddingsURL: URL
    public let modelIdentity: String
    public let version: String
    public let encoderSHA256: String
    public let compileCacheDirectory: URL

    public init(
        encoderURL: URL,
        speciesTableURL: URL,
        embeddingsURL: URL,
        modelIdentity: String,
        version: String,
        encoderSHA256: String,
        compileCacheDirectory: URL
    ) {
        self.encoderURL = encoderURL
        self.speciesTableURL = speciesTableURL
        self.embeddingsURL = embeddingsURL
        self.modelIdentity = modelIdentity
        self.version = version
        self.encoderSHA256 = encoderSHA256.lowercased()
        self.compileCacheDirectory = compileCacheDirectory
    }
}

public enum SpeciesClassifierError: Error, Equatable, LocalizedError {
    case missingArtifact
    case invalidArtifactIdentity
    case invalidSpeciesTable
    case unsupportedEmbeddingType(String)
    case invalidEmbeddingTableLength
    case modelCompilationFailed
    case invalidModelInput
    case invalidModelOutput
    case imageDecodeFailed
    case imagePreprocessingFailed
    case predictionFailed

    public var errorDescription: String? {
        switch self {
        case .missingArtifact: "A required species model file is missing."
        case .invalidArtifactIdentity: "The species model identity is invalid."
        case .invalidSpeciesTable: "The species table is invalid."
        case let .unsupportedEmbeddingType(type): "Unsupported embedding type: \(type)."
        case .invalidEmbeddingTableLength: "The species embedding table has the wrong size."
        case .modelCompilationFailed: "The species model could not be compiled."
        case .invalidModelInput: "The species model has an unexpected input."
        case .invalidModelOutput: "The species model has an unexpected output."
        case .imageDecodeFailed: "The selected image could not be decoded."
        case .imagePreprocessingFailed: "The selected image could not be prepared."
        case .predictionFailed: "Species recognition failed."
        }
    }
}

public struct SpeciesResult: Equatable, Sendable {
    public let rank: Int
    public let name: String
    public let scientificName: String
    public let matchScore: Float
    public let danger: String?
    public let dangerNote: String?
}

public actor SpeciesClassifier {
    private let model: MLModel
    private let table: EmbeddingTable
    private let inputSide = 224
    private let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    private let std: [Float] = [0.26862954, 0.26130258, 0.27577711]

    private init(model: MLModel, table: EmbeddingTable) {
        self.model = model
        self.table = table
    }

    public static func load(
        artifacts: SpeciesArtifactSet,
        computeUnits: MLComputeUnits = .all
    ) async throws -> SpeciesClassifier {
        guard !artifacts.modelIdentity.isEmpty,
              !artifacts.version.isEmpty,
              artifacts.encoderSHA256.count == 64,
              artifacts.encoderSHA256.allSatisfy(\.isHexDigit),
              FileManager.default.fileExists(atPath: artifacts.encoderURL.path)
        else {
            throw SpeciesClassifierError.invalidArtifactIdentity
        }
        let table = try EmbeddingTable(
            speciesTableURL: artifacts.speciesTableURL,
            embeddingsURL: artifacts.embeddingsURL
        )
        guard table.dim == 768 else {
            throw SpeciesClassifierError.invalidSpeciesTable
        }
        let compiledURL = try await compiledModelURL(for: artifacts)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model: MLModel
        do {
            model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw SpeciesClassifierError.modelCompilationFailed
        }
        try validate(model: model)
        return SpeciesClassifier(model: model, table: table)
    }

    public var speciesCount: Int { table.species.count }

    public func classify(_ imageData: Data, topK: Int = 3) async throws -> [SpeciesResult] {
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw SpeciesClassifierError.imageDecodeFailed
        }
        return try classifySynchronously(image, topK: topK)
    }

    public func classify(_ image: CGImage, topK: Int = 3) async throws -> [SpeciesResult] {
        try classifySynchronously(image, topK: topK)
    }

    private func classifySynchronously(
        _ image: CGImage,
        topK: Int
    ) throws -> [SpeciesResult] {
        guard topK > 0 else { return [] }
        let input = try preprocess(image)
        let features: MLFeatureProvider
        do {
            features = try model.prediction(from: MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(multiArray: input)]
            ))
        } catch {
            throw SpeciesClassifierError.predictionFailed
        }
        guard let multi = features.featureValue(for: "embedding")?.multiArrayValue,
              multi.count == table.dim else {
            throw SpeciesClassifierError.invalidModelOutput
        }
        var embedding = [Float](repeating: 0, count: table.dim)
        for index in embedding.indices { embedding[index] = multi[index].floatValue }
        return rank(embedding, topK: topK)
    }

    public func rank(_ embedding: [Float], topK: Int = 3) -> [SpeciesResult] {
        guard embedding.count == table.dim, topK > 0 else { return [] }
        let count = table.species.count
        var cosine = [Float](repeating: 0, count: count)
        for index in 0..<count { cosine[index] = table.cosine(embedding, row: index) }
        let maximum = cosine.max() ?? 0
        var denominator: Float = 0
        var exponentials = [Float](repeating: 0, count: count)
        for index in 0..<count {
            exponentials[index] = exp(100 * (cosine[index] - maximum))
            denominator += exponentials[index]
        }
        let order = (0..<count).sorted {
            cosine[$0] == cosine[$1] ? $0 < $1 : cosine[$0] > cosine[$1]
        }
        return order.prefix(min(topK, count)).enumerated().map { rank, index in
            let species = table.species[index]
            return SpeciesResult(
                rank: rank + 1,
                name: species.name,
                scientificName: species.sci,
                matchScore: exponentials[index] / denominator,
                danger: species.danger,
                dangerNote: species.dangerNote
            )
        }
    }

    private static func compiledModelURL(for artifacts: SpeciesArtifactSet) async throws -> URL {
        let fileManager = FileManager.default
        let destination = artifacts.compileCacheDirectory
            .appendingPathComponent(artifacts.encoderSHA256, isDirectory: true)
            .appendingPathExtension("mlmodelc")
        if fileManager.fileExists(atPath: destination.path) { return destination }
        let partial = artifacts.compileCacheDirectory
            .appendingPathComponent(
                "\(artifacts.encoderSHA256).\(UUID().uuidString).partial",
                isDirectory: true
            )
            .appendingPathExtension("mlmodelc")
        do {
            try fileManager.createDirectory(
                at: artifacts.compileCacheDirectory,
                withIntermediateDirectories: true
            )
            let temporary = try await MLModel.compileModel(at: artifacts.encoderURL)
            if fileManager.fileExists(atPath: destination.path) {
                try? fileManager.removeItem(at: temporary)
                return destination
            }
            try fileManager.moveItem(at: temporary, to: partial)
            if fileManager.fileExists(atPath: destination.path) {
                try? fileManager.removeItem(at: partial)
                return destination
            }
            try fileManager.moveItem(at: partial, to: destination)
            return destination
        } catch {
            try? fileManager.removeItem(at: partial)
            if fileManager.fileExists(atPath: destination.path) {
                return destination
            }
            throw SpeciesClassifierError.modelCompilationFailed
        }
    }

    private static func validate(model: MLModel) throws {
        guard let input = model.modelDescription.inputDescriptionsByName["image"],
              input.type == .multiArray,
              input.multiArrayConstraint?.shape.map(\.intValue) == [1, 3, 224, 224]
        else { throw SpeciesClassifierError.invalidModelInput }
        guard let output = model.modelDescription.outputDescriptionsByName["embedding"],
              output.type == .multiArray,
              output.multiArrayConstraint?.shape.map(\.intValue) == [1, 768]
        else { throw SpeciesClassifierError.invalidModelOutput }
    }

    private func preprocess(_ image: CGImage) throws -> MLMultiArray {
        guard let format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
        ), let source = try? vImage_Buffer(cgImage: image, format: format) else {
            throw SpeciesClassifierError.imagePreprocessingFailed
        }
        let sourceBuffer = source
        defer { free(sourceBuffer.data) }
        let scale = CGFloat(inputSide) / CGFloat(min(image.width, image.height))
        // torchvision's integer Resize truncates the scaled long edge.
        let width = max(inputSide, Int(CGFloat(image.width) * scale))
        let height = max(inputSide, Int(CGFloat(image.height) * scale))
        let offsetX = (width - inputSide) / 2
        let offsetY = (height - inputSide) / 2
        let horizontal = Self.bicubicCoefficients(
            inputSize: image.width,
            outputSize: width,
            outputRange: offsetX..<(offsetX + inputSide)
        )
        let vertical = Self.bicubicCoefficients(
            inputSize: image.height,
            outputSize: height,
            outputRange: offsetY..<(offsetY + inputSide)
        )
        let sourcePixels = sourceBuffer.data.assumingMemoryBound(to: UInt8.self)
        var intermediate = [UInt8](
            repeating: 0,
            count: image.height * inputSide * 3
        )
        for y in 0..<image.height {
            let sourceRow = sourcePixels.advanced(by: y * sourceBuffer.rowBytes)
            for x in 0..<inputSide {
                let coefficients = horizontal[x]
                for channel in 0..<3 {
                    var sum = 1 << 21
                    for index in coefficients.weights.indices {
                        sum += Int(sourceRow[(coefficients.start + index) * 4 + channel + 1])
                            * coefficients.weights[index]
                    }
                    intermediate[(y * inputSide + x) * 3 + channel] =
                        UInt8(clamping: sum >> 22)
                }
            }
        }
        let output = try MLMultiArray(
            shape: [1, 3, NSNumber(value: inputSide), NSNumber(value: inputSide)],
            dataType: .float32
        )
        output.withUnsafeMutableBytes { raw, _ in
            let floats = raw.bindMemory(to: Float.self)
            for y in 0..<inputSide {
                let coefficients = vertical[y]
                for x in 0..<inputSide {
                    for channel in 0..<3 {
                        var sum = 1 << 21
                        for index in coefficients.weights.indices {
                            sum += Int(intermediate[
                                ((coefficients.start + index) * inputSide + x) * 3 + channel
                            ]) * coefficients.weights[index]
                        }
                        let value = Float(UInt8(clamping: sum >> 22)) / 255
                        floats[channel * inputSide * inputSide + y * inputSide + x] =
                            (value - mean[channel]) / std[channel]
                    }
                }
            }
        }
        return output
    }

    private struct ResampleCoefficients {
        let start: Int
        let weights: [Int]
    }

    private static func bicubicCoefficients(
        inputSize: Int,
        outputSize: Int,
        outputRange: Range<Int>
    ) -> [ResampleCoefficients] {
        let scale = Double(inputSize) / Double(outputSize)
        let filterScale = max(1, scale)
        let support = 2 * filterScale
        return outputRange.map { outputIndex in
            let center = (Double(outputIndex) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5))
            let end = min(inputSize, Int(center + support + 0.5))
            var values = (start..<end).map { inputIndex in
                bicubic((Double(inputIndex) - center + 0.5) / filterScale)
            }
            let total = values.reduce(0, +)
            if total != 0 {
                values = values.map { $0 / total }
            }
            return ResampleCoefficients(
                start: start,
                weights: values.map {
                    Int($0 < 0 ? -0.5 + $0 * Double(1 << 22) : 0.5 + $0 * Double(1 << 22))
                }
            )
        }
    }

    private static func bicubic(_ value: Double) -> Double {
        let x = abs(value)
        if x < 1 {
            return (1.5 * x - 2.5) * x * x + 1
        }
        if x < 2 {
            return (((x - 5) * x + 8) * x - 4) * -0.5
        }
        return 0
    }
}
