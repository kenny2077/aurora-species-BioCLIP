# Contributing

Thanks for your interest in improving the Aurora project!

## Ground rules

1. **Measured claims only.** This project's whole point is the benchmark
   record. Any change that affects model behavior ships with a scorecard run
   (`exam_brancha_v3.py` / `run_exam.py`) against the sealed exam — before and
   after, same fixture, temp 0.
2. **Sealed stays sealed.** Never train on `exam_s1_sealed.json`,
   `exam_v3_sealed.json`, `exam-iphone13/` photos, or the
   `expert_user_language.json` alias[0] phrasings. Contaminated benchmarks are
   worthless benchmarks.
3. **Reviewed sources only.** Training data comes from reviewed corpora. If
   you add sources, document provenance and license.

## Workflow

1. Fork, branch (`feat/...` or `fix/...`).
2. Make your change (data recipe, training, exam, Swift kit).
3. Run the relevant exam; include the scorecard diff in your PR.
4. PRs are reviewed against: reproducibility (scripts run as documented),
   benchmark honesty (no fixture edits), and license cleanliness.

## Code style

Python: stdlib-first, type hints, no required heavy deps outside the training
path. Swift: match AuroraSpeciesKit's existing structure.
