import Accelerate
import CoreML
import Foundation

public struct SpeciesResult: Sendable {
    public let name: String
    public let sci: String
    public let score: Float       // softmax probability over the 504-class table
    public let danger: String?    // "high" / "medium" when flagged
}

/// On-device species classification: one ViT-L/14 forward pass through the
/// fp16 Core ML encoder (Neural Engine on A15+), then cosine against the
/// 504x768 embedding table. Mathematically identical to full BioCLIP-2
/// zero-shot — the text tower is precomputed into the table.
public final class SpeciesClassifier {
    private let model: MLModel
    private let table: EmbeddingTable
    private let mean: [Float]
    private let std: [Float]
    private let inputSide = 224

    /// - Parameters:
    ///   - computeUnits: `.all` routes to the Neural Engine on A15+; `.cpuOnly`
    ///     is the deterministic fallback for parity debugging.
    public init(bundle: Bundle = .main, computeUnits: MLComputeUnits = .all) throws {
        guard let modelURL = bundle.url(forResource: "BioCLIP2-ImageEncoder", withExtension: "mlmodelc")
        else {
            throw NSError(domain: "AuroraSpeciesKit", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "BioCLIP2-ImageEncoder.mlmodelc not in bundle"])
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: config)
        table = try EmbeddingTable(bundle: bundle)
        // CLIP normalization constants baked at conversion time (coreml_meta.json values)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    }

    public var speciesCount: Int { table.species.count }

    /// Classify a JPEG/HEIC/PNG. Returns the top-K species sorted by probability.
    public func classify(_ image: CGImage, topK: Int = 5) throws -> [SpeciesResult] {
        let input = try preprocess(image)
        let features = try model.prediction(from: MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(multiArray: input)]))
        guard let multi = features.featureValue(for: "embedding")?.multiArrayValue else {
            throw NSError(domain: "AuroraSpeciesKit", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "no 'embedding' output from encoder"])
        }
        var emb = [Float](repeating: 0, count: table.dim)
        for j in 0..<table.dim { emb[j] = multi[j].floatValue }
        return rank(emb, topK: topK)
    }

    /// Cosine-rank a raw (unnormalized) embedding against the table.
    public func rank(_ embedding: [Float], topK: Int = 5) -> [SpeciesResult] {
        let n = table.species.count
        var cos = [Float](repeating: 0, count: n)
        for i in 0..<n { cos[i] = table.cosine(embedding, row: i) }
        // softmax over the standard 100*cos temperature used by CLIP zero-shot
        var maxc: Float = -.infinity
        for c in cos { maxc = max(maxc, c) }
        var sum: Float = 0
        var exps = [Float](repeating: 0, count: n)
        for i in 0..<n { exps[i] = exp(100 * (cos[i] - maxc)); sum += exps[i] }
        let order = (0..<n).sorted { cos[$0] > cos[$1] }
        return order.prefix(topK).map { i in
            SpeciesResult(name: table.species[i].name, sci: table.species[i].sci,
                          score: exps[i] / sum, danger: table.species[i].danger)
        }
    }

    // MARK: - Preprocess (must match open_clip: resize shorter side 224 bicubic,
    // center crop 224, /255, (x-mean)/std)

    private func preprocess(_ image: CGImage) throws -> MLMultiArray {
        guard let src = vImage_CGImageFormat(bitsPerComponent: 8, bitsPerPixel: 32,
                                             colorSpace: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue))
                .flatMap({ try? vImage_Buffer(cgImage: image, format: $0) })
        else {
            throw NSError(domain: "AuroraSpeciesKit", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "cannot decode image into vImage_Buffer"])
        }
        defer { free(src.data) }
        let side = CGFloat(inputSide)
        let scale = side / CGFloat(min(image.width, image.height))
        var scaledW = UInt(round(CGFloat(image.width) * scale))
        var scaledH = UInt(round(CGFloat(image.height) * scale))
        scaledW = max(scaledW, UInt(inputSide)); scaledH = max(scaledH, UInt(inputSide))
        var dst = try vImage_Buffer(width: Int(scaledW), height: Int(scaledH),
                                    bitsPerPixel: src.bitsPerPixel)
        defer { free(dst.data) }
        let err = vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
        guard err == kvImageNoError else {
            throw NSError(domain: "AuroraSpeciesKit", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "vImageScale error \(err)"])
        }
        // center crop 224x224
        let offX = (Int(scaledW) - inputSide) / 2
        let offY = (Int(scaledH) - inputSide) / 2
        var cropped = vImage_Buffer(data: dst.data.advanced(by: offY * Int(dst.rowBytes) + offX * 4),
                                    height: vImagePixelCount(inputSide), width: vImagePixelCount(inputSide),
                                    rowBytes: dst.rowBytes)

        let out = try MLMultiArray(shape: [1, 3, NSNumber(value: inputSide), NSNumber(value: inputSide)],
                                   dataType: .float32)
        let H = inputSide, W = inputSide
        let pixels = cropped.data.assumingMemoryBound(to: UInt8.self)  // ARGB (noneSkipFirst, big-endian byte order BGRA on LE? byte 0=A when alphaInfo noneSkipFirst)
        let rowBytes = cropped.rowBytes
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let f = raw.bindMemory(to: Float.self)
            for y in 0..<H {
                let row = pixels.advanced(by: y * rowBytes)
                for x in 0..<W {
                    let p = row.advanced(by: x * 4)
                    // noneSkipFirst => bytes are [A, R, G, B]
                    let rgb = [Float(p[1]) / 255, Float(p[2]) / 255, Float(p[3]) / 255]
                    for c in 0..<3 {
                        f[c * H * W + y * W + x] = (rgb[c] - mean[c]) / std[c]
                    }
                }
            }
        }
        _ = cropped  // borrowed view into dst
        return out
    }
}
