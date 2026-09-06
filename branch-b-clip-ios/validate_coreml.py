#!/usr/bin/env python3
"""validate_coreml.py — parity gates for the converted encoder.

Gate 1 (any OS):  torch fp32 vs torch fp16  — expected drift of the fp16 export.
Gate 2 (macOS):   torch fp32 vs Core ML      — the real acceptance gate; the
                  on-device exam (iPhone 13) must then reproduce it.

Predictions over the Aurora-500 table must agree on top-1 within a fraction of
a percent of cosine similarity, and per-photo top-1 agreement should be 100%
on a clean photo set (report anything less).

  python validate_coreml.py --images <dir or photo.jpg ...>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: PLC0415
import torch  # noqa: PLC0415

ROOT = Path(__file__).parent
MODEL = "hf-hub:imageomics/bioclip-2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_table() -> tuple[torch.Tensor, list[dict]]:
    tbl = json.loads((ROOT / "species_table.json").read_text(encoding="utf-8"))
    mat = np.fromfile(ROOT / "species_embeddings.f16.bin", dtype=np.float16)
    mat = mat.reshape(tbl["count"], tbl["dim"]).astype(np.float32)
    return torch.from_numpy(mat), tbl["species"]


def load_model_and_preprocess():
    import open_clip  # noqa: PLC0415
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    except Exception as exc:
        if "weights_only" in str(exc).lower() or "Weights" in str(exc):
            import functools  # noqa: PLC0415
            torch.load = functools.partial(torch.load, weights_only=False)
            model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
        else:
            raise
    return model.eval(), preprocess


def collect(images: list[Path]) -> list[Path]:
    out = []
    for p in images:
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]
        elif p.exists():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", type=Path, required=True)
    ap.add_argument("--max", type=int, default=20, help="photos to check")
    args = ap.parse_args()
    from PIL import Image  # noqa: PLC0415

    table, species = load_table()
    model, preprocess = load_model_and_preprocess()
    import copy  # noqa: PLC0415
    model_h = copy.deepcopy(model).half()  # fp16 pass on its own copy (no in-place mutation)
    photos = collect(args.images)[: args.max]
    if not photos:
        raise SystemExit("no photos found")

    rows, agree16 = [], []
    coreml_ok = coreml_model = None
    try:  # Gate 2 only exists on macOS
        import coremltools as ct  # noqa: PLC0415
        coreml_model = ct.models.MLModel(str(ROOT / "coreml" / "BioCLIP2-ImageEncoder.mlpackage"))
        coreml_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"[gate2] Core ML predict unavailable here ({exc.__class__.__name__}) — macOS only")

    for p in photos:
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            f32 = model.encode_image(img)[0]
            f32 = f32 / f32.norm()
            f16 = model_h.encode_image(img.half())[0].float()
            f16 = f16 / f16.norm()
        cos16 = float(f32 @ f16)
        top16 = int(((table @ f16).argmax()))
        top32 = int(((table @ f32).argmax()))
        agree16.append(top16 == top32)
        row = {"image": str(p), "cos_fp32_vs_fp16": round(cos16, 6),
               "top1_fp32": species[top32]["sci"], "top1_fp16": species[top16]["sci"]}
        if coreml_ok:
            out = coreml_model.predict({"image": img.numpy().astype(np.float32)})
            emb = torch.from_numpy(np.array(out["embedding"], dtype=np.float32))[0]
            emb = emb / emb.norm()
            row["cos_fp32_vs_coreml"] = round(float(f32 @ emb), 6)
            row["top1_coreml"] = species[int((table @ emb).argmax())]["sci"]
        rows.append(row)
        print(json.dumps(row))

    report = {
        "photos": len(rows), "top1_agreement_fp32_vs_fp16": f"{sum(agree16)}/{len(agree16)}",
        "min_cos_fp32_vs_fp16": min(r["cos_fp32_vs_fp16"] for r in rows),
        "coreml_checked": bool(coreml_ok),
        "min_cos_fp32_vs_coreml": (min((r["cos_fp32_vs_coreml"] for r in rows
                                        if "cos_fp32_vs_coreml" in r), default=None)),
        "rows": rows,
    }
    (ROOT / "coreml" / "validation_report.json").write_text(json.dumps(report, indent=1),
                                                            encoding="utf-8")
    print(f"\n[validate] fp32↔fp16 top-1 agreement {sum(agree16)}/{len(agree16)}, "
          f"min cos {report['min_cos_fp32_vs_fp16']:.6f} -> coreml/validation_report.json")


if __name__ == "__main__":
    main()
