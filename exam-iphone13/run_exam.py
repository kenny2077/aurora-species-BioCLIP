#!/usr/bin/env python3
"""run_exam.py — PC-side reference exam of the species-ID pipeline (iPhone-faithful).

The classification branch over the Aurora-500 embedding table. The iPhone 13 Core ML package
must reproduce THIS scorecard on-device (gate: top-1 agreement = 100%, cosine
delta < ~1e-3). Optional --fusion adds Branch A: the CLIP top-5 shortlist goes
into a list-constrained prompt for the LoRA-served VLM, which answers with the
species + field marks + survival protocol — the shipped product behavior.

  python run_exam.py                                   # Branch B only
  python run_exam.py --fusion http://127.0.0.1:8123    # + VLM fusion stage
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
MODEL = "hf-hub:imageomics/bioclip-2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_table():
    tbl = json.loads((ROOT.parent / "species-classifier" / "species_table.json").read_text(encoding="utf-8"))
    import numpy as np  # noqa: PLC0415
    mat = np.fromfile(ROOT.parent / "species-classifier" / "species_embeddings.f16.bin",
                      dtype=np.float16).reshape(tbl["count"], tbl["dim"]).astype("float32")
    return tbl, mat


def encode_images(photos):
    import open_clip  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    except Exception as exc:
        if "weights_only" in str(exc).lower() or "Weights" in str(exc):
            import functools  # noqa: PLC0415
            torch.load = functools.partial(torch.load, weights_only=False)
            model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
        else:
            raise
    model = model.to(device).eval()
    feats = []
    with torch.no_grad():
        for p in photos:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(img)[0]
            feats.append((f / f.norm()).float().cpu())
    return torch.stack(feats)  # [P, 768]


def vlm_fusion(url: str, image: Path, names: list[str]) -> dict:
    """List-constrained confirm over the LoRA-served VLM (FOCI-style, as in lora.py quiz)."""
    import requests  # noqa: PLC0415
    b64 = base64.b64encode(image.read_bytes()).decode()
    prompt = ("What animal is in this photo? Answer with exactly one name from this list: "
              + ", ".join(names) + ". Then one sentence on key field marks and one sentence "
              "on what to do if this animal is near your camp.")
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt}]}], "temperature": 0.0, "max_tokens": 300}
    resp = requests.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    return {"answer": resp.json()["choices"][0]["message"]["content"].strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=ROOT / "photos")
    ap.add_argument("--fusion", default=None, help="llama-server URL, e.g. http://127.0.0.1:8123")
    ap.add_argument("--tag", default=None, help="results filename tag")
    args = ap.parse_args()

    tbl, mat = load_table()
    species = tbl["species"]
    labels = json.loads((ROOT / "photo_labels.json").read_text(encoding="utf-8"))
    name2sci = {s["name"]: s["sci"] for s in species}

    folders = sorted(d for d in args.photos.iterdir() if d.is_dir())
    photos, truths = [], []
    for d in folders:
        for img in sorted(d.glob("*")):
            if img.suffix.lower() in IMG_EXTS:
                photos.append(img)
                truths.append(labels.get(d.name, name2sci.get(d.name, d.name)))
    if not photos:
        raise SystemExit("no photos found")
    print(f"[exam] {len(photos)} photos, {len(folders)} species, "
          f"table {len(species)} classes | fusion={'yes' if args.fusion else 'no'}")

    t0 = time.time()
    feats = encode_images(photos)
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    logits = 100.0 * feats @ torch.from_numpy(mat).T  # [P, N]
    probs = logits.softmax(dim=-1)
    top = torch.topk(logits, k=5, dim=-1)
    ms = (time.time() - t0) * 1000 / len(photos)

    rows, hit = [], {1: 0, 3: 0, 5: 0}
    for i, p in enumerate(photos):
        idx = top.indices[i].tolist()
        pr = [float(probs[i, j]) for j in idx]
        names = [species[j]["name"] for j in idx]
        scis = [species[j]["sci"] for j in idx]
        true_sci = truths[i]
        hits = [j for j, s in enumerate(scis) if s == true_sci]
        for k in (1, 3, 5):
            hit[k] += bool(hits and hits[0] < k)
        row = {"image": str(p), "true": true_sci,
               "top5": [{"name": names[j], "sci": scis[j], "p": round(pr[j], 4)}
                        for j in range(5)],
               "hit": hits[0] if hits else None}
        if args.fusion:
            try:
                ans = vlm_fusion(args.fusion, p, names)
                low = ans["answer"].lower()
                pick = next((scis[j] for j in range(5)
                             if names[j].lower() in low), scis[0])
                row["fusion_pick"] = pick
                row["fusion_ok"] = pick == true_sci
                row["fusion_answer"] = ans["answer"]
            except Exception as exc:  # noqa: BLE001
                row["fusion_error"] = exc.__class__.__name__
        rows.append(row)

    n = len(rows)
    fusion_hit = sum(r.get("fusion_ok", False) for r in rows)
    fusion_used = sum(1 for r in rows if "fusion_ok" in r)
    score = {
        "model": MODEL, "table_classes": len(species), "photos": n,
        "top1": f"{hit[1]}/{n} = {hit[1]/n*100:.1f}%",
        "top3": f"{hit[3]}/{n} = {hit[3]/n*100:.1f}%",
        "top5": f"{hit[5]}/{n} = {hit[5]/n*100:.1f}%",
        "fusion": (f"{fusion_hit}/{fusion_used} = {fusion_hit/fusion_used*100:.1f}%"
                   if fusion_used else None),
        "ms_per_photo": round(ms, 1),
        "misses_top1": [{"image": r["image"], "true": r["true"],
                         "pred": r["top5"][0]} for r in rows
                        if not (r["hit"] == 0)],
    }
    tag = args.tag or ("fusion" if args.fusion else "branchB")
    out = ROOT / "results" / f"exam_pc_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"score": score, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(score, indent=1))
    print(f"[exam] -> {out}")


if __name__ == "__main__":
    main()
