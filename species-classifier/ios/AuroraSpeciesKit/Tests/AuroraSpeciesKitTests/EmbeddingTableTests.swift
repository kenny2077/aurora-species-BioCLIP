import Foundation
import XCTest
@testable import AuroraSpeciesKit

final class EmbeddingTableTests: XCTestCase {
    func testLoadsMatchingTableAndEmbeddings() throws {
        let urls = try fixture(dim: 2, speciesCount: 2, halfValues: [0x3c00, 0, 0, 0x3c00])
        let table = try EmbeddingTable(
            speciesTableURL: urls.table,
            embeddingsURL: urls.embeddings
        )

        XCTAssertEqual(table.dim, 2)
        XCTAssertEqual(table.species.map(\.sci), ["A a", "B b"])
        XCTAssertEqual(table.cosine([1, 0], row: 0), 1, accuracy: 0.0001)
        XCTAssertEqual(table.cosine([1, 0], row: 1), 0, accuracy: 0.0001)
    }

    func testRejectsTruncatedEmbeddingTable() throws {
        let urls = try fixture(dim: 2, speciesCount: 2, halfValues: [0x3c00])

        XCTAssertThrowsError(try EmbeddingTable(
            speciesTableURL: urls.table,
            embeddingsURL: urls.embeddings
        )) { error in
            XCTAssertEqual(error as? SpeciesClassifierError, .invalidEmbeddingTableLength)
        }
    }

    func testRejectsSpeciesCountMismatch() throws {
        let urls = try fixture(dim: 2, speciesCount: 3, halfValues: Array(repeating: 0, count: 6))

        XCTAssertThrowsError(try EmbeddingTable(
            speciesTableURL: urls.table,
            embeddingsURL: urls.embeddings
        )) { error in
            XCTAssertEqual(error as? SpeciesClassifierError, .invalidSpeciesTable)
        }
    }

    private func fixture(
        dim: Int,
        speciesCount: Int,
        halfValues: [UInt16]
    ) throws -> (table: URL, embeddings: URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let tableURL = directory.appendingPathComponent("species_table.json")
        let embeddingsURL = directory.appendingPathComponent("species_embeddings.f16.bin")
        let species: [[String: Any]] = [
            ["name": "Alpha", "sci": "A a", "group": "bird"],
            ["name": "Beta", "sci": "B b", "group": "mammal"],
        ]
        let metadata: [String: Any] = [
            "format": "aurora-species-embeddings-v1",
            "model": "test",
            "dtype": "float16",
            "dim": dim,
            "count": speciesCount,
            "species": species,
        ]
        try JSONSerialization.data(withJSONObject: metadata).write(to: tableURL)
        let data = halfValues.withUnsafeBytes { Data($0) }
        try data.write(to: embeddingsURL)
        return (tableURL, embeddingsURL)
    }
}
