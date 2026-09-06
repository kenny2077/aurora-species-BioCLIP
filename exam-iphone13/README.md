# exam-iphone13/ — the species-recognition exam for the iPhone 13

The full pipeline exam, run three ways: **PC reference** (this machine),
**Mac validation** (fp32 vs Core ML parity), and **on-device** (the iPhone 13
itself). All three must agree before shipping.

## The sealed pool

`photos/` = 43 species folders, 85 photos:
- **37 of the original sealed 39** (2 quarantined — see below), plus
- **46 fresh CC-licensed iNaturalist photos** for 23 additional Aurora-500
  species chosen for coverage: megafauna (moose, elk, wolf, bobcat, beaver,
  otter), common birds (robin, cardinal, jay, hawk, vulture, woodpecker, owl,
  goldfinch), venomous reptiles (copperhead, timber + western diamond
  rattlesnake, alligator), box turtle, and 4 amphibians.

`photo_labels.json` maps folder → scientific name (labels never leak into the
classifier — folders are only the answer key).

**Quarantine (`photos_quarantine/reasons.json`)** — label/scene failures caught
by the model itself, worth keeping as pipeline lessons:
1. "eastern chipmunk" photo was a **barred owl holding a chipmunk** — model
   said owl (0.92); the iNat observation taxon was right but the scene's
   primary subject wasn't.
2. "coyote" photo was a coyote walking **in front of an elephant seal colony** —
   seals dominate the frame.
3. "moose" photo was iNat's **content-warning placeholder**, not the photo.

Lesson for the fetcher: taxon verification is necessary but not sufficient —
add a "primary subject is the taxon" check (CLIP sanity check against the
label at fetch time catches all three).

## PC reference scorecards (results/)

Branch B, BioCLIP-2 zero-shot over the **504-class Aurora table**, iPhone-faithful
inputs (224 center-crop), RTX 4050, ~190 ms/photo:

| Cut | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Full pool (85 photos, 43 species) | **92.9% (79/85)** | 98.8% | **100%** |
| Fresh 46 photos | 97.9% | 100% | 100% |
| Sealed Worker1 37 photos (under 504 classes) | 86.5% | — | — |
| (memory lane: same sealed set under 20 classes) | 97.4% | — | — |

All six top-1 misses are hard lookalikes/hard shots and every one stays inside
the top-5: juvenile bald eagle→red-tailed hawk, cinnamon-morph black
bear→brown bear, blue jay→california scrub-jay, cottontail→swamp rabbit, the
known pale desert-morph red fox→pronghorn, skunk→ringtail.

**Fusion finding (important):** letting the VLM re-pick from the CLIP top-5
*lowers* accuracy to 74.1% (broke 18 correct CLIP answers, fixed 2). Margin
gating never beats CLIP-alone either (best 92.9%). **Ship policy: Branch B
decides the species; Branch A never re-classifies — it explains the decided
species and applies survival doctrine.** (Quantified in
`results/exam_pc_fusion.json` + the policy table in the session log.)

## Running the exam

```bash
python run_exam.py                                  # Branch B -> results/exam_pc_branchB.json
python run_exam.py --fusion http://127.0.0.1:8123   # + VLM stage (llama-server from branch-a-survival)
```

## On-device protocol (iPhone 13, iOS 17+)

1. On the Mac: compile the encoder, copy resources into AuroraSpeciesKit
   (commands in `../branch-b-clip-ios/README.md`), copy `photos/` +
   `photo_labels.json` into `Tests/AuroraSpeciesKitTests/Resources/Photos/`.
2. `xcodebuild test -destination 'platform=iOS,id=<iphone13-udid>'` — the
   `ExamTests` suite runs the full pool on-device and prints the scorecard +
   mean ms/photo; it also writes `exam_device_results.json` to the app sandbox.
3. Gates to pass vs this folder's PC reference:
   - top-1 agreement with PC reference: **no photo flips classes** (cosine
     drift from fp16 + ANE is ~1e-3; the 0.38-margin misses are far away)
   - mean latency: record it (ANE expectation: low tens of ms/photo — measure,
     don't promise), peak memory < ~1.5 GB for the classifier alone.
4. Record the numbers in `results/exam_device_iphone13.json` (keep the same
   schema as the PC run for diffing).
