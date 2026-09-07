import Foundation
import ImageIO

/// The iPhone-13 species-recognition exam, runnable on-device.
///
/// Expects in the test bundle (Tests/AuroraSpeciesKitTests/Resources/Photos):
///   <species folder>/<photo>.jpg   (folder name = species display name)
///   photo_labels.json              {"<folder>": "<scientific name>", ...}
/// Produces the same scorecard as the PC reference run (run_exam.py):
/// top-1/top-3/top-5 over the 504-class Aurora table.
public struct AuroraExamRunner {
    public struct PhotoCase: Sendable {
        public let folder: String
        public let url: URL
        public let trueSci: String
    }

    public struct Scorecard: Sendable {
        public let photos: Int
        public let top1: Int
        public let top3: Int
        public let top5: Int
        public let meanMsPerPhoto: Double
        public var summary: String {
            String(format: "iPhone exam: %d/%d top-1 (%.1f%%) | top-3 %.1f%% | top-5 %.1f%% | %.0f ms/photo",
                   top1, photos, Double(top1) / Double(photos) * 100,
                   Double(top3) / Double(photos) * 100,
                   Double(top5) / Double(photos) * 100, meanMsPerPhoto)
        }
    }

    let labels: [String: String]

    public init(labelsURL: URL) throws {
        labels = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: labelsURL)
        )
    }

    /// Collect photo cases from a bundle directory named "Photos".
    public static func collectCases(root: URL) throws -> [PhotoCase] {
        let labels = try JSONDecoder().decode([String: String].self,
                                              from: Data(contentsOf: root.appendingPathComponent("photo_labels.json")))
        let exts = ["jpg", "jpeg", "png"]
        var cases: [PhotoCase] = []
        let dirs = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey])
        for dir in dirs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where dir.hasDirectoryPath {
            let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            for f in files where exts.contains(f.pathExtension.lowercased()) {
                let folder = dir.lastPathComponent
                cases.append(PhotoCase(folder: folder, url: f, trueSci: labels[folder] ?? folder))
            }
        }
        return cases
    }

    public func run(
        classifier: SpeciesClassifier,
        cases: [PhotoCase]
    ) async throws -> (Scorecard, [Row]) {
        var top1 = 0, top3 = 0, top5 = 0
        var totalMs: Double = 0
        var rows: [Row] = []
        for c in cases {
            guard let src = CGImageSourceCreateWithURL(c.url as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
            else { continue }
            let t0 = Date()
            let results = try await classifier.classify(img, topK: 5)
            totalMs += -t0.timeIntervalSinceNow * 1000
            let rank = results.firstIndex { $0.scientificName == c.trueSci }
            if let r = rank {
                if r < 1 { top1 += 1 }
                if r < 3 { top3 += 1 }
                if r < 5 { top5 += 1 }
            }
            rows.append(Row(folder: c.folder, image: c.url.lastPathComponent, trueSci: c.trueSci,
                            predicted: results.first?.scientificName ?? "?", rank: rank.map { Int64($0) }))
        }
        let n = max(rows.count, 1)
        let card = Scorecard(photos: rows.count, top1: top1, top3: top3, top5: top5,
                             meanMsPerPhoto: totalMs / Double(n))
        return (card, rows)
    }

    public struct Row: Codable, Sendable {
        let folder: String
        let image: String
        let trueSci: String
        let predicted: String
        let rank: Int64?
    }
}
