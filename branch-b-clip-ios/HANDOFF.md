# HANDOFF — branch-b-clip-ios (the iOS BioCLIP package)

**Audience:** the agent/engineer on the MacBook who will finish the iPhone 13
exam. Written 2026-09-05 by the Windows-side agent after the full PC-side
build. Everything here is measured or verified on this machine; the few
unverified items are marked **[unmeasured]**. No prior chat context is needed.

---

## 0. TL;DR — your job on the Mac (~1 hour)

1. Compile the encoder: `xcrun coremlcompiler compile coreml/BioCLIP2-ImageEncoder.mlpackage coreml/`
2. Gate 2 parity: `python validate_coreml.py --images ../exam-iphone13/photos/ --max 20`
   (needs `coremltools` in a Mac venv; gate 1 already passes — see §5).
3. Load resources into the Swift package and run the on-device exam on the
   iPhone 13 (§7, §8).
4. Acceptance: **no photo flips species class vs the PC reference, top-5
   stays 100%**, record ANE latency + peak memory into
   `../exam-iphone13/results/exam_device_iphone13.json`.

If a gate fails, read §10 (failure playbook) before touching anything.

---

## 1. Mission & lineage (why this folder exists)

**Product:** Aurora — offline wildlife species ID + survival doctrine for
North America. Two-branch architecture, empirically settled by Worker1 and
re-confirmed by the exam in this package:

- **Branch B — Perceive & Classify (this folder):** BioCLIP-2 zero-shot.
  CLIP **decides the species**, full stop.
- **Branch A — Know & Explain (`../branch-a-survival/`):** Qwen3-VL-2B-2B +
  survival-knowledge LoRA. It **never re-classifies** — it explains the
  decided species and applies survival doctrine.

The scorecard history that proves the split (same sealed photos each time):

| Model | Classes | Score |
|---|---|---|
| BioCLIP-2 zero-shot | 20 | 38/39 = **97.4%** (Worker1) |
| Base VLM, list-constrained | 20 | 82% (Worker1) |
| VLM + LoRA, list-constrained | 20 | 79% (Worker1) |
| **BioCLIP-2 zero-shot** | **504** | **92.9% top-1 / 100% top-5** (this package) |
| Naive fusion (VLM re-picks from CLIP top-5) | 504 | **74.1% — WORSE; rejected** |

The fusion number matters: on this exam the VLM broke 18 correct CLIP answers
and fixed 2. Margin-gating never beat CLIP-alone either. **Do not "improve"
the pipeline by letting the VLM vote.** The VLM's output contract is:
given the decided species + photo → field marks + survival protocol.

## 2. What is in this folder

| Path | Size | Origin |
|---|---|---|
| `coreml/BioCLIP2-ImageEncoder.mlpackage` | 581 MB | **generated** — fp16 Core ML ViT-L/14 image encoder (built in WSL2) |
| `coreml/coreml_meta.json` | — | input/output shapes + CLIP normalization constants |
| `coreml/validation_report.json` | — | Gate-1 results (fp32↔fp16) |
| `species_embeddings.f16.bin` | 0.77 MB | **generated** — 504×768 fp16, row i = L2-normalized mean text embedding of species i |
| `species_table.json` | — | names/sci/group/danger + format metadata; **row order = .bin row order** |
| `build_embeddings.py` | — | rebuilds the table (source of truth for prompts) |
| `convert_coreml.py` | — | rebuilds the mlpackage (Linux/macOS) |
| `validate_coreml.py` | — | Gate 1 (any OS) + Gate 2 (macOS only) |
| `build/Dockerfile` | — | pinned isolated conversion env (torch 2.7.0 + coremltools 9) |
| `ios/AuroraSpeciesKit/` | — | Swift package: `EmbeddingTable`, `SpeciesClassifier`, `AuroraExamRunner` + `ExamTests` |
| `manifest.json` | — | sha256 of everything (bin/*.exe/*.dll excluded — pinned by upstream tag) |
| `README.md`, `HANDOFF.md` | — | docs |

Resources that must be added on the Mac before the Swift build (not here yet
because Windows can't compile mlmodelc):
`BioCLIP2-ImageEncoder.mlmodelc` → `Sources/AuroraSpeciesKit/Resources/`,
plus copies of `species_table.json` and `species_embeddings.f16.bin`.
Exam assets go in the **test** bundle: `../exam-iphone13/photos/` (85 photos,
43 folders) + `photo_labels.json` → `Tests/AuroraSpeciesKitTests/Resources/Photos/`.

## 3. Architecture — why "no text tower" is exact, not approximate

Zero-shot classification is: image → 768-d vector; each species name → 768-d
vector; prediction = largest cosine. The species vectors depend **only on the
prompt text** — never on the photo — so they are precomputed on the PC with
BioCLIP-2's own text encoder and shipped as the 0.77 MB table. The phone does
one ViT forward pass + 504 dot products. This is the same math with the
constant part hoisted at build time; running the full model on-device would
produce identical logits plus 0.9 GB of dead weight.

On-device flow (per photo):

```
CGImage → vImage lanczos resize (shorter side → 224) → center crop 224
        → normalize (x/255 − mean)/std   [mean/std in coreml_meta.json + Swift]
        → MLModel "image" [1,3,224,224] fp32 in → "embedding" [1,768] out (fp16 compute, ANE)
        → L2 normalize → cosine vs 504 rows → softmax(100·cos) → top-K + confidence
```

Softmax temperature 100·cos matches `clip.py` and all PC scoring — keep it.

## 4. Invariants — do not silently change these

1. **Prompt templates are part of the weights.** The three templates in
   `build_embeddings.py` must stay byte-identical to `clip.py`'s `TEMPLATES`.
   Changing wording changes every embedding and invalidates all scores.
2. **Row order coupling:** `species_embeddings.f16.bin` row i ↔
   `species_table.json species[i]`. The Swift reader indexes rows by `i*dim`.
3. **Identity is the scientific name.** Common names diverge between sources
   (exam folder "grizzly bear" = table "brown bear"/*Ursus arctos*; "wapiti" =
   elk). All scoring matches sci names via `photo_labels.json`.
4. **CLIP decides; the VLM explains.** See §1 — quantified.
5. **The exam pool is sealed.** Never train/tune on `../exam-iphone13/photos/`.
   The training workbench (scripts + photo provenance) is `../training/`; all
   39 original sealed photos survive in the exam pool (37) + quarantine (2).
6. **Licenses:** only CC0/CC-BY/CC-BY-SA photos ship; every download is logged
   in `../exam-iphone13/provenance.csv`.
7. **Softmax temperature 100**, same as every scorecard to date.
8. Growing the species list (§9) must never require re-converting the encoder
   — only the table changes.

## 5. Measured state of the world (2026-09-05)

- **PC reference exam** (`../exam-iphone13/results/exam_pc_branchB.json`):
  85 photos / 43 species / 504 classes → **top-1 79/85 (92.9%), top-3 98.8%,
  top-5 85/85 (100%)**, ~190 ms/photo (RTX 4050 GPU — phone latency is a
  different story, measure it).
- **Subset split:** fresh 48 photos 97.9% top-1; Worker1-sealed 37 photos
  86.5% under 504 classes (they scored 97.4% under 20 — lookalike dilution).
- **The 6 top-1 misses** (all remain in top-5): juvenile bald eagle→red-tailed
  hawk (0.38), cinnamon-morph black bear→brown bear (0.52), blue
  jay→california scrub-jay (0.96!), cottontail→swamp rabbit (0.65), pale
  desert-morph red fox→pronghorn (0.32, Worker1's original single miss),
  skunk→ringtail (0.34). Low margin ⇔ lookalike: the UI should surface top-3
  when margin is thin rather than pretending certainty.
- **Gate 1 (fp32↔fp16 torch):** min cosine 0.999987, top-1 agreement 6/6
  (`coreml/validation_report.json`).
- **Gate 2 (fp32↔Core ML): pending — your Mac run.** Accept: per-photo cosine
  ≥ 0.999, top-1 agreement 100%. coremltools **cannot execute** models on
  Linux/Windows; that's why it's on you.
- **Naive fusion:** 74.1% (63/85) — documented in
  `../exam-iphone13/results/exam_pc_fusion.json`; rejected (§1).
- **Quarantined exam photos** (`../exam-iphone13/photos_quarantine/reasons.json`):
  (a) "eastern chipmunk" = a barred owl holding a chipmunk (model right, label
  wrong), (b) "coyote" = coyote in front of an elephant-seal colony, (c) "moose"
  = iNat's content-warning placeholder image. Lesson: taxon verification is not
  subject verification — a fetch-time CLIP sanity check against the label
  catches all three. Worth adding before the next expansion.
- **Sibling server:** start Branch A with `../branch-a-survival/serve.ps1`
  (llama.cpp **b10819** win-cuda, Qwen3-VL-2B Q4_K_M + mmproj-F16 +
  `animal-lora.gguf`, port 8123, `--jinja -ngl 99`).

## 6. Environment & build gotchas (all hit for real — with fixes)

1. **iNaturalist API:** the filter is `iconic_taxa` (plural). `iconic_taxon_name`
   is **silently ignored** → your "birds" query returns plants and insects.
   Also: `quality_grade=research`, `captive=false`, place ids US=1, CA=6712;
   ~1 req/s rate limit; verify the observation's taxon name starts with the
   requested sci name (iNat's `taxon_name` search is loose).
2. **torch fused attention is not CoreML-convertible** — trace fails with
   `_native_multi_head_attention not implemented`. Fix (in
   `convert_coreml.py`): `torch.backends.mha.set_fastpath_enabled(False)` AND
   monkeypatch `F.scaled_dot_product_attention` to plain
   matmul/softmax/add (function included there — copy it, don't re-derive).
3. **coremltools 9:** `ct.TensorType(dtype=...)` takes **numpy** dtypes
   (`np.float32`), not `torch.float32`.
4. **Version pin is load-bearing:** torch 2.14 + its torchvision →
   `operator torchvision::nms does not exist` at open_clip import. The tested
   pair is **torch 2.7.0 + torchvision 0.22.0 + coremltools ≥ 9.0** (the
   Dockerfile pins it; the WSL venv `~/venvs/coreml` uses it).
5. **open_clip checkpoint loading** predates torch `weights_only` default →
   wrap `create_model_and_transforms` with the `torch.load` partial
   (`weights_only=False`) fallback — already in every script here.
6. **HF cache reuse:** the 1.7 GB BioCLIP-2 weights live in the Windows cache;
   WSL reuses them via `HF_HOME=/mnt/c/Users/heath/.cache/huggingface`.
7. **Windows python** used by all PC scripts: 3.12.10, open_clip 3.3.0,
   torch 2.6.0+cu124 (RTX 4050). Scripts avoid fancy typing; keep it that way.
8. **This repo sits in OneDrive.** Don't create Python venvs on the /mnt/c
   path from WSL (9P is glacial) — use `~/venvs/` in WSL. Expect sync churn
   after big writes (the mlpackage took a while to settle).
9. **`nn.Module.half()` mutates in place** — for fp16 parity checks deepcopy
   the model first (`validate_coreml.py` does this correctly now).
10. **llama.cpp release tags:** the GitHub "latest" release has no binaries;
    binaries live under weekly `b<NNNN>` tags (b10819 pinned here).

## 7. Mac runbook (detailed)

```bash
cd <repo>/species-id/branch-b-clip-ios

# 1. compile (SPM cannot compile .mlpackage; Xcode-coremlcompiler can)
xcrun coremlcompiler compile coreml/BioCLIP2-ImageEncoder.mlpackage coreml/
#    -> coreml/BioCLIP2-ImageEncoder.mlmodelc

# 2. resources into the library bundle
cp coreml/BioCLIP2-ImageEncoder.mlmodelc species_table.json species_embeddings.f16.bin \
   ios/AuroraSpeciesKit/Sources/AuroraSpeciesKit/Resources/

# 3. Gate 2 — the acceptance gate that Windows cannot run
python3 -m venv ~/venvs/coreml && source ~/venvs/coreml/bin/activate
pip install "coremltools>=9.0" open_clip_torch "numpy<2" pillow
python validate_coreml.py --images ../exam-iphone13/photos --max 20
#    pass: every "cos_fp32_vs_coreml" ≥ 0.999, top1_coreml == top1_fp32 always

# 4. exam assets into the test bundle
mkdir -p ios/AuroraSpeciesKit/Tests/AuroraSpeciesKitTests/Resources
cp -r ../exam-iphone13/photos ios/AuroraSpeciesKit/Tests/AuroraSpeciesKitTests/Resources/Photos
cp ../exam-iphone13/photo_labels.json ios/AuroraSpeciesKit/Tests/AuroraSpeciesKitTests/Resources/Photos/

# 5. device run (iPhone 13, iOS 17+, Developer Mode on)
cd ios/AuroraSpeciesKit
xcodebuild test -scheme AuroraSpeciesKit \
  -destination 'platform=iOS,id=<IPHONE13_UDID>'
# ExamTests.testFullPoolExam prints the scorecard + writes exam_device_results.json
```

## 8. iPhone 13 acceptance gates

| Gate | Requirement | Why |
|---|---|---|
| G1 fp32↔fp16 | done (§5) | precision drift baseline |
| G2 fp32↔Core ML (Mac) | cosine ≥ 0.999, 0 class flips on 20 photos | conversion parity |
| G3 on-device vs PC | **0 class flips across all 85** (top-1 ≥ 91.8%, i.e. ≤1 flip tolerated if G2 marginal — record which), top-5 = 100% | the actual exam |
| G4 perf | record mean ms/photo + peak RSS | A15 ANE expectation: low tens of ms **[unmeasured — do not promise]** |
| G5 memory | classifier resident < ~1.5 GB | 4 GB device budget |

Hardware facts for the report: iPhone 13 = A15 Bionic (16-core ANE), 4 GB RAM.
Encoder 581 MB fp16 → ANE. Table 0.77 MB. Branch A does NOT fit alongside on
the base model (~2.2 GB more) — fusion on the 13 is sequential or paired-device.

## 9. Rebuilding / growing artifacts (what regenerates what)

- **Species list** → `../species/build_species_list.py` (edit `QUOTAS`, e.g.
  `{"Aves": 350, "Mammalia": 200, "Reptilia": 120, "Amphibia": 60}` for ~750–1000;
  dangerous species are overlay-ensured). Emits `../species/aurora-500.json`
  (rename the `name` field if you grow it, or it will lie).
- **Embedding table** → `python build_embeddings.py` (PC). **The encoder does
  NOT need re-conversion when only the list grows** — swap the new
  `species_table.json` + `.bin` into the bundle and you're done. This is the
  whole point of the architecture.
- **Core ML encoder** → `python convert_coreml.py` (WSL2/Docker/macOS), only
  when the model itself changes (it hasn't since b10819-era build).
- **Exam photos** → `../exam-iphone13/fetch_exam_photos.py` (resumable; add
  species to `EXAM24_SCI`-style list). Add the fetch-time subject check (§5)
  first if you can.

## 10. Failure playbook

- **G2 cosine slightly low (0.998–0.999) but class flips rare:** first check
  preprocessing equality (resize interpolation is the usual suspect — Swift
  uses `kvImageHighQualityResampling` vs PIL bicubic; try
  `CILanczosScaleTransform` if drift is marginal). fp16 compute inside CoreML
  also differs slightly from torch fp16 — that's what G3's tolerance covers.
- **G3 flips on lookalike pairs:** the six known misses carry top-1
  confidence 0.32–0.96 (five below 0.70); if drift-induced flips appear they
  should concentrate there. Re-check G2 before blaming the phone.
- **ANE fallbacks, in order:** `computeUnits = .cpuAndGPU` (fp16 GPU), int8
  palettization via `coremltools.optimize` (re-run G2), or float32 CPU —
  each trades latency/memory, never ship without recording both.
- **`operator torchvision::nms does not exist`** or any open_clip import
  weirdness → wrong torch/torchvision pair; use the pin (§6.4).
- **ExamTests can't find Photos/** → resources must be under
  `Tests/AuroraSpeciesKitTests/Resources/Photos/` with `photo_labels.json`
  INSIDE that folder (the Package.swift `.copy("Photos")` expects the dir).

## 11. Context beyond this folder

- `../README.md` — umbrella: both packages + the 504-species
  decision (why not 1000) + the embeddings-only rationale.
- `../exam-iphone13/README.md` — full exam protocol, PC scorecards, quarantine
  log, on-device protocol.
- `../branch-a-survival/` — Part 1 (2B + LoRA + knowledge), serve scripts,
  the fusion prompt contract (`../exam-iphone13/run_exam.py:vlm_fusion`).
- `../species/` — the list builder + danger overlay (36 danger-flagged
  species: all NA venomous snakes, bears, cougar, gila monster, widows,
  recluse, bark scorpions, ticks). `SpeciesResult.danger` surfaces this.
- `../training/` — Worker1's original two-branch workbench (`clip.py`,
  `lora.py`, `kb_build.py`, `animals.json`, photo provenance); the original
  sealed set lives in `../exam-iphone13/photos/` + `photos_quarantine/`.
- **Signing:** Aurora's trust flow is Ed25519 + SHA-256 over `manifest.json`
  at catalog time. Manifests here are generated unsigned — wire the existing
  Aurora signing flow over them, don't invent a new one.

*End of handoff. If you change any invariant in §4, update this file first —
it is the contract every future agent reads.*
