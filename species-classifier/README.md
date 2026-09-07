# species-classifier/ — PART 2: the iOS BioCLIP package

> **Mac-side agent? Start with [HANDOFF.md](HANDOFF.md)** — architecture,
> invariants, measured state, every build gotcha with its fix, and the
> step-by-step runbook to finish the iPhone 13 exam.

**Runs BioCLIP-2 species recognition natively on iPhone (A15+): a fp16 Core ML
image encoder + the 504-species embedding table. No text tower ships.**

| Artifact | Size | Role |
|---|---|---|
| `coreml/BioCLIP2-ImageEncoder.mlpackage` | ~0.6 GB fp16 | ViT-L/14 image encoder, photo → 768-d embedding |
| `species_embeddings.f16.bin` | 0.77 MB | 504 × 768 fp16 text-embedding table (precomputed) |
| `species_table.json` | — | names/sci/group/danger + format metadata (row order = .bin rows) |
| `ios/AuroraSpeciesKit/` | — | Swift package: URL-loaded classifier + table reader + on-device exam |
| `build_embeddings.py` | — | rebuild the table for a new species list |
| `convert_coreml.py` | — | rebuild the mlpackage (Linux/macOS) |
| `validate_coreml.py` | — | parity gates: fp32↔fp16 (any OS), fp32↔Core ML (macOS) |

## Pipeline on device (per photo, ~tens of ms on ANE)

```
photo → resize 224 (shorter side, bicubic) → center crop → normalize (CLIP mean/std)
      → Core ML ViT-L/14 (Neural Engine) → 768-d embedding → normalize
      → cosine vs 504 rows → softmax(100·cos) → top-5 species + confidence
```

Identical math to full BioCLIP-2 zero-shot — the species vectors depend only
on prompt text, so precomputing them loses nothing (see `../README.md`).

## Rebuild steps

```bash
# 1. species list -> embeddings (PC, uses the same templates as clip.py)
python build_embeddings.py

# 2. image encoder -> Core ML (WSL2 Ubuntu works; isolated Dockerfile below)
python convert_coreml.py        # -> coreml/BioCLIP2-ImageEncoder.mlpackage

# 3. on a Mac: validate the portable package before publishing it
python validate_coreml.py --images ../exam-iphone13/photos/ --max 20
```

The Swift package contains no model resources. An app supplies verified file
URLs in `SpeciesArtifactSet`; `SpeciesClassifier.load` compiles the downloaded
model package off the main thread and caches the `.mlmodelc` by encoder digest.

## Isolated conversion image (reproducible builds)

```bash
docker build -t aurora-coreml-build build/
docker run --rm -v "$PWD:/pkg" -e HF_HOME=/hf-cache aurora-coreml-build
```

Pins torch 2.7.0 + torchvision 0.22.0 (the coremltools-tested pair) and
coremltools 9 — the version matrix matters (see `build/Dockerfile`).

## iPhone 13 exam gates (acceptance)

1. `validate_coreml.py` on the Mac: fp32↔Core ML cosine ≥ 0.999 per photo,
   top-1 agreement 100% on the checked photos.
2. On-device XCTest (`ExamTests.swift`) on the iPhone 13: top-1 within a point
   of the PC reference (92.9%), top-5 = 100%, and record ms/photo + peak
   memory. Protocol: `../exam-iphone13/README.md`.

## Known conversion notes (hit in practice, fixed in `convert_coreml.py`)

- torch's fused attention (`_native_multi_head_attention` / SDPA fastpath) is
  not convertible — the script disables the MHA fastpath and decomposes SDPA
  to plain matmul/softmax before tracing.
- `ct.TensorType(dtype=...)` takes **numpy** dtypes in coremltools 9.
- Linux cannot execute MLModel predictions — `validate_coreml.py` gate 2 needs
  a Mac; everything else is cross-platform.
