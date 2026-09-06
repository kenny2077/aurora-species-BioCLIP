<p align="center">
  <h1 align="center">Aurora Species ID 🐦</h1>
  <p align="center"><strong>On-device wildlife species identification — BioCLIP-2, 504 species, one image forward pass.</strong></p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-8A5CF6?style=for-the-badge&labelColor=111827" alt="MIT License"></a>
  <a href="https://huggingface.co/imageomics/BioCLIP-2"><img src="https://img.shields.io/badge/Model-BioCLIP--2-67E8F9?style=for-the-badge&labelColor=111827" alt="BioCLIP-2"></a>
  <img src="https://img.shields.io/badge/Target-iPhone%2013%2B%20(ANE)-F472B6?style=for-the-badge&labelColor=111827" alt="Platform">
  <img src="https://img.shields.io/badge/Classes-504%20species-34D399?style=for-the-badge&labelColor=111827" alt="Species">
</p>

---

**Aurora Species ID** converts [BioCLIP-2](https://huggingface.co/imageomics/BioCLIP-2)
into a fully on-device species classifier for iOS: the image encoder is traced to
a Core ML package, the text tower is replaced by a **precomputed 504-species
embedding table** (0.77 MB), and a native Swift package runs the whole thing on
the Neural Engine. One ViT forward pass plus a dot product — no network, no
text encoder, no server.

Validated on a sealed, quarantined exam: **92.9% top-1 / 98.8% top-3 / 100%
top-5** across 504 classes on 85 labeled photos from 43 species.

## Benchmark

| Metric (sealed exam, 85 photos / 43 species / 504 classes) | Score |
|---|---|
| Top-1 | **92.9%** |
| Top-3 | **98.8%** |
| Top-5 | **100%** |

Every top-1 miss stayed inside the top-5, and all misses were genuine hard
cases (juvenile plumages, color morphs, sibling species). Three exam photos
were quarantined with written reasons during label auditing.

## How it works

```
photo ──► BioCLIP-2 image encoder (Core ML, fp16, Neural Engine)
                │  (one 256-token ViT pass, ~581 MB fp16 weights)
                ▼
        256-d image embedding
                │  dot product
                ▼
        504-species embedding table (0.77 MB fp16, replaces the text tower)
                ▼
        top-1 + top-5 + confidence  ──►  downstream survival guidance
```

With a fixed class list, the text encoder never ships: the 504 text embeddings
are precomputed once on a PC and loaded as a plain binary table on device.

## Repository layout

```
├── species-classifier/     Core ML encoder + table + native Swift runtime
│   ├── convert_coreml.py     trace BioCLIP-2 → Core ML fp16 (WSL2/macOS)
│   ├── build_embeddings.py   build the 504-species embedding table
│   ├── validate_coreml.py    fp32↔fp16 parity gate (Gate 1) + Core ML gate (Gate 2, Mac)
│   ├── ios/AuroraSpeciesKit/ native Swift classifier + on-device exam runner (XCTest)
│   └── manifest.json         sha256 pinning for every file in the package
├── species/               Aurora-500 species list builder (iNat prevalence derived)
├── exam-iphone13/         sealed exam: 85 photos, labels, provenance, scorecards
└── make_manifest.py       reproducible package manifest generator
```

Heavy binaries (the 581 MB `.mlpackage`, base model weights) are gitignored but
pinned by sha256 in each package's `manifest.json`.

## Quick start

**Rebuild the species table (PC, Python 3.11+, CUDA optional):**

```bash
python species-classifier/build_embeddings.py
```

**Convert the encoder to Core ML** (WSL2 with `torch==2.7.0` +
`coremltools==9.0`, pinned in the Dockerfile; see
`species-classifier/HANDOFF.md`), then run the parity gate:

```bash
python species-classifier/validate_coreml.py
```

**iOS:** open `species-classifier/ios/AuroraSpeciesKit/` in Xcode, drop in the
compiled `.mlmodelc` and `species_embeddings.f16.bin`, and run the exam XCTest
on device.

## The species list: Aurora-500

The 504 classes are **derived from data, not opinion** — the most-observed
species per taxon group in iNaturalist research-grade wild observations across
the US and Canada (250 birds, 120 mammals, 70 reptiles, 40 amphibians) plus 24
safety-ensured dangerous species. 36 species carry a danger flag in
`species/danger_overlay.json`. Growing the list is a script re-run plus a
table rebuild.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Please keep
photo additions CC-licensed and recorded in `exam-iphone13/provenance.csv`.

## License

Code: [MIT](LICENSE). Model weights and photos have their own licenses — see
Third-party models below.

## Acknowledgments

- [BioCLIP-2](https://huggingface.co/imageomics/BioCLIP-2) (Imageomics Institute) — the vision backbone
- [iNaturalist](https://www.inaturalist.org) contributors — every exam and training photo is CC-licensed and attributed in `exam-iphone13/provenance.csv`
- [Apple coremltools](https://github.com/apple/coremltools) and [llama.cpp](https://github.com/ggml-org/llama.cpp) — conversion and on-device runtime
- Part of the [Aurora Survival LoRA](https://github.com/kenny2077/aurora-survival-lora) offline survival assistant
