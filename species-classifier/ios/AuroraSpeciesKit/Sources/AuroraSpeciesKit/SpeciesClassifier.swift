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

public struct SpeciesInferenceMetrics: Equatable, Sendable {
    public let decodeMilliseconds: Double
    public let preprocessingMilliseconds: Double
    public let predictionMilliseconds: Double
    public let rankingMilliseconds: Double
    public let totalMilliseconds: Double
}

public struct SpeciesClassificationOutput: Equatable, Sendable {
    public let results: [SpeciesResult]
    public let metrics: SpeciesInferenceMetrics
}

public actor SpeciesClassifier {
    private let model: MLModel
    private let table: EmbeddingTable

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
        try classifyMeasured(imageData, topK: topK).results
    }

    public func classifyMeasured(
        _ imageData: Data,
        topK: Int = 3
    ) throws -> SpeciesClassificationOutput {
        let totalStarted = CFAbsoluteTimeGetCurrent()
        let decodeStarted = CFAbsoluteTimeGetCurrent()
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw SpeciesClassifierError.imageDecodeFailed
        }
        return try classifySynchronously(
            image,
            topK: topK,
            decodeMilliseconds: Self.elapsedMilliseconds(since: decodeStarted),
            totalStarted: totalStarted
        )
    }

    public func classify(_ image: CGImage, topK: Int = 3) async throws -> [SpeciesResult] {
        try classifySynchronously(
            image,
            topK: topK,
            decodeMilliseconds: 0,
            totalStarted: CFAbsoluteTimeGetCurrent()
        ).results
    }

    private func classifySynchronously(
        _ image: CGImage,
        topK: Int,
        decodeMilliseconds: Double,
        totalStarted: CFAbsoluteTime
    ) throws -> SpeciesClassificationOutput {
        let preprocessingStarted = CFAbsoluteTimeGetCurrent()
        let input = try Self.preprocessProduction(image)
        let preprocessingMilliseconds = Self.elapsedMilliseconds(since: preprocessingStarted)
        let predictionStarted = CFAbsoluteTimeGetCurrent()
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
        let predictionMilliseconds = Self.elapsedMilliseconds(since: predictionStarted)
        var embedding = [Float](repeating: 0, count: table.dim)
        for index in embedding.indices { embedding[index] = multi[index].floatValue }
        let rankingStarted = CFAbsoluteTimeGetCurrent()
        let results = rank(embedding, topK: topK)
        return SpeciesClassificationOutput(
            results: results,
            metrics: SpeciesInferenceMetrics(
                decodeMilliseconds: decodeMilliseconds,
                preprocessingMilliseconds: preprocessingMilliseconds,
                predictionMilliseconds: predictionMilliseconds,
                rankingMilliseconds: Self.elapsedMilliseconds(since: rankingStarted),
                totalMilliseconds: Self.elapsedMilliseconds(since: totalStarted)
            )
        )
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

    static func resizedDimensions(width: Int, height: Int) -> (width: Int, height: Int) {
        let scale = CGFloat(224) / CGFloat(min(width, height))
        return (
            max(224, Int(CGFloat(width) * scale)),
            max(224, Int(CGFloat(height) * scale))
        )
    }

    static func preprocessProduction(_ image: CGImage) throws -> MLMultiArray {
        guard let format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
        ), let source = try? vImage_Buffer(cgImage: image, format: format) else {
            throw SpeciesClassifierError.imagePreprocessingFailed
        }
        var sourceBuffer = source
        defer { free(sourceBuffer.data) }
        let dimensions = resizedDimensions(width: image.width, height: image.height)
        guard var destination = try? vImage_Buffer(
            width: dimensions.width,
            height: dimensions.height,
            bitsPerPixel: 32
        ) else { throw SpeciesClassifierError.imagePreprocessingFailed }
        defer { free(destination.data) }
        guard vImageScale_ARGB8888(
            &sourceBuffer,
            &destination,
            nil,
            vImage_Flags(kvImageHighQualityResampling)
        ) == kvImageNoError else {
            throw SpeciesClassifierError.imagePreprocessingFailed
        }
        let offsetX = (dimensions.width - 224) / 2
        let offsetY = (dimensions.height - 224) / 2
        let pixels = destination.data
            .advanced(by: offsetY * destination.rowBytes + offsetX * 4)
            .assumingMemoryBound(to: UInt8.self)
        let output = try MLMultiArray(
            shape: [1, 3, 224, 224],
            dataType: .float32
        )
        output.withUnsafeMutableBytes { raw, _ in
            let floats = raw.bindMemory(to: Float.self)
            for y in 0..<224 {
                let row = pixels.advanced(by: y * destination.rowBytes)
                for x in 0..<224 {
                    let pixel = row.advanced(by: x * 4)
                    for channel in 0..<3 {
                        let value = Float(pixel[channel + 1]) / 255
                        floats[channel * 224 * 224 + y * 224 + x] =
                            (value - mean[channel]) / std[channel]
                    }
                }
            }
        }
        return output
    }

    static func preprocessReferenceExact(_ image: CGImage) throws -> MLMultiArray {
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
        let dimensions = resizedDimensions(width: image.width, height: image.height)
        let width = dimensions.width
        let height = dimensions.height
        let offsetX = (width - 224) / 2
        let offsetY = (height - 224) / 2
        let horizontal = Self.bicubicCoefficients(
            inputSize: image.width,
            outputSize: width,
            outputRange: offsetX..<(offsetX + 224)
        )
        let vertical = Self.bicubicCoefficients(
            inputSize: image.height,
            outputSize: height,
            outputRange: offsetY..<(offsetY + 224)
        )
        let sourcePixels = sourceBuffer.data.assumingMemoryBound(to: UInt8.self)
        var intermediate = [UInt8](
            repeating: 0,
            count: image.height * 224 * 3
        )
        for y in 0..<image.height {
            let sourceRow = sourcePixels.advanced(by: y * sourceBuffer.rowBytes)
            for x in 0..<224 {
                let coefficients = horizontal[x]
                for channel in 0..<3 {
                    var sum = 1 << 21
                    for index in coefficients.weights.indices {
                        sum += Int(sourceRow[(coefficients.start + index) * 4 + channel + 1])
                            * coefficients.weights[index]
                    }
                    intermediate[(y * 224 + x) * 3 + channel] =
                        UInt8(clamping: sum >> 22)
                }
            }
        }
        let output = try MLMultiArray(
            shape: [1, 3, 224, 224],
            dataType: .float32
        )
        output.withUnsafeMutableBytes { raw, _ in
            let floats = raw.bindMemory(to: Float.self)
            for y in 0..<224 {
                let coefficients = vertical[y]
                for x in 0..<224 {
                    for channel in 0..<3 {
                        var sum = 1 << 21
                        for index in coefficients.weights.indices {
                            sum += Int(intermediate[
                                ((coefficients.start + index) * 224 + x) * 3 + channel
                            ]) * coefficients.weights[index]
                        }
                        let value = Float(UInt8(clamping: sum >> 22)) / 255
                        floats[channel * 224 * 224 + y * 224 + x] =
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

    private static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    private static let std: [Float] = [0.26862954, 0.26130258, 0.27577711]

    private static func elapsedMilliseconds(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1_000
    }
}
