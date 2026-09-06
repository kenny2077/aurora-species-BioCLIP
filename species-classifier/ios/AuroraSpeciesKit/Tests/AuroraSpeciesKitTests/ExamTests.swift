import XCTest
@testable import AuroraSpeciesKit

/// The iPhone 13 species-recognition exam.
///
/// Requires in this test bundle: `Photos/` (species folders with photos +
/// `photo_labels.json`) and in AuroraSpeciesKit's bundle: the compiled
/// `BioCLIP2-ImageEncoder.mlmodelc`, `species_table.json`, and
/// `species_embeddings.f16.bin`. See ../../exam-iphone13/README.md.
final class ExamTests: XCTestCase {
    func testFullPoolExam() throws {
        let bundle = Bundle(for: ExamTests.self)
        let cases = try AuroraExamRunner.collectCases(bundle: bundle)
        XCTAssertGreaterThan(cases.count, 40, "exam pool missing — copy photos/ into the test bundle")

        let classifier = try SpeciesClassifier(bundle: Bundle(for: SpeciesClassifier.self))
        let runner = try AuroraExamRunner(bundle: bundle)
        let (card, rows) = try runner.run(classifier: classifier, cases: cases)

        print("\(card.summary)")
        print("misses:", rows.filter { ($0.rank ?? 99) != 0 }.map { "\($0.folder)/\($0.image) -> rank \($0.rank ?? -1)" })

        // Record the same schema as the PC run for diffing.
        let result: [String: Any] = ["photos": card.photos,
                                     "top1": card.top1,
                                     "top3": card.top3,
                                     "top5": card.top5,
                                     "ms_per_photo": card.meanMsPerPhoto,
                                     "rows": rows.map { ["folder": $0.folder, "image": $0.image,
                                                         "true": $0.trueSci, "predicted": $0.predicted,
                                                         "rank": $0.rank ?? -1] }]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("exam_device_results.json")
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: url)
        print("results -> \(url.path)")

        // Acceptance gates vs the PC reference (92.9% top-1 / 100% top-5):
        // allow ONE class flip from fp16/ANE drift; top-5 must stay perfect.
        let top1Frac = Double(card.top1) / Double(card.photos)
        XCTAssertGreaterThan(top1Frac, 0.90, "top-1 dropped more than one flip below PC reference")
        XCTAssertEqual(card.top5, card.photos, "top-5 must be 100% (PC reference)")
    }
}
