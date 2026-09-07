import Foundation

public struct SpeciesEntry: Decodable, Equatable, Sendable {
    public let name: String
    public let sci: String
    public let group: String
    public let count: Int?
    public let rank: Int?
    public let danger: String?
    public let dangerNote: String?

    enum CodingKeys: String, CodingKey {
        case name, sci, group, count, rank, danger
        case dangerNote = "danger_note"
    }
}

struct TableMeta: Decodable {
    let format: String
    let model: String
    let dtype: String
    let dim: Int
    let count: Int
    let species: [SpeciesEntry]
}

/// The 504x768 fp16 species text-embedding table (0.77 MB) — the ship-ready
/// replacement for BioCLIP-2's text tower. Row i is the L2-normalized mean
/// CLIP text embedding of species[i] over the standard prompt templates;
/// cosine similarity to an image embedding IS the zero-shot logit.
public final class EmbeddingTable {
    public let dim: Int
    public let species: [SpeciesEntry]
    /// row-major [count x dim], fp32, L2-normalized rows (~1.5 MB in RAM)
    public let rows: [Float]

    public init(speciesTableURL: URL, embeddingsURL: URL) throws {
        guard FileManager.default.fileExists(atPath: speciesTableURL.path),
              FileManager.default.fileExists(atPath: embeddingsURL.path) else {
            throw SpeciesClassifierError.missingArtifact
        }
        let meta: TableMeta
        do {
            meta = try JSONDecoder().decode(
                TableMeta.self,
                from: Data(contentsOf: speciesTableURL)
            )
        } catch {
            throw SpeciesClassifierError.invalidSpeciesTable
        }
        guard meta.dtype == "float16" else {
            throw SpeciesClassifierError.unsupportedEmbeddingType(meta.dtype)
        }
        guard meta.dim > 0, meta.count > 0, meta.species.count == meta.count else {
            throw SpeciesClassifierError.invalidSpeciesTable
        }
        let raw = try Data(contentsOf: embeddingsURL, options: .mappedIfSafe)
        guard raw.count == meta.count * meta.dim * MemoryLayout<UInt16>.size else {
            throw SpeciesClassifierError.invalidEmbeddingTableLength
        }
        var rows = [Float](repeating: 0, count: meta.count * meta.dim)
        raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            let half = buf.bindMemory(to: UInt16.self)
            for i in 0..<rows.count {
                rows[i] = Float(Float16(bitPattern: half[i]))
            }
        }
        dim = meta.dim
        species = meta.species
        self.rows = rows
    }

    /// L2-normalized row i (rows are already normalized at build time).
    public func cosine(_ embedding: [Float], row i: Int) -> Float {
        let base = i * dim
        var dot: Float = 0
        var nrm: Float = 0
        for j in 0..<dim {
            dot += embedding[j] * rows[base + j]
            nrm += embedding[j] * embedding[j]
        }
        return dot / sqrt(nrm)
    }
}
