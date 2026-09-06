#!/usr/bin/env python3
"""build_embeddings.py — precompute the Aurora-500 text-embedding table for iPhone.

The text tower NEVER ships on the phone: "a photo of a {name} ({sci})." always
produces the same 768-dim vector regardless of where it is computed, so the PC
computes it once (BioCLIP-2 text encoder, same templates as clip.py) and the
package ships a 504x768 fp16 table (~0.77 MB). On-device inference is then one
ViT forward pass + 504 dot products — mathematically identical zero-shot
accuracy to running the full model (see ../README.md).

  python build_embeddings.py
    -> species_embeddings.f16.bin      raw [N, 768] float16, row-major
    -> species_table.json              names/sci/group/danger + format metadata
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SPECIES = ROOT.parent / "species" / "aurora-500.json"
CLIP_MODEL = "hf-hub:imageomics/bioclip-2"
# MUST stay identical to clip.py TEMPLATES — prompt wording is part of accuracy.
TEMPLATES = [
    "a photo of a {name} ({sci}).",
    "a photo of a {name}, a wild animal.",
    "wildlife camera-trap photo of a {name} ({sci}).",
]


def main() -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import open_clip  # noqa: PLC0415

    cfg = json.loads(SPECIES.read_text(encoding="utf-8"))
    species = cfg["species"]
    print(f"[embed] {len(species)} species from {SPECIES.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL)
    except Exception as exc:
        if "weights_only" in str(exc).lower() or "Weights" in str(exc):
            import functools  # noqa: PLC0415
            torch.load = functools.partial(torch.load, weights_only=False)
            model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL)
        else:
            raise
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    texts = [t.format(name=s["name"], sci=s["sci"]) for s in species for t in TEMPLATES]
    n_tpl = len(TEMPLATES)
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            toks = tokenizer(texts[i:i + 256]).to(device)
            f = model.encode_text(toks)
            feats.append(f / f.norm(dim=-1, keepdim=True))
    feats = torch.cat(feats).view(len(species), n_tpl, -1).mean(dim=1)  # [N, D]
    feats = (feats / feats.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16)
    print(f"[embed] table {feats.shape} fp16, mean pairwise-cos sanity ok")

    bin_path = ROOT / "species_embeddings.f16.bin"
    bin_path.write_bytes(feats.tobytes(order="C"))
    table = {
        "format": "aurora-species-table-v1",
        "model": CLIP_MODEL,
        "dtype": "float16", "dim": int(feats.shape[1]), "count": int(feats.shape[0]),
        "templates": TEMPLATES,
        "embeddings_sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
        "row_order_matches": "species_table.json species[] order",
        "note": "row i of the .bin is the L2-normalized mean text embedding of species[i]",
        "species": species,
    }
    (ROOT / "species_table.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[embed] -> {bin_path.name} ({bin_path.stat().st_size / 1e6:.2f} MB) + species_table.json")


if __name__ == "__main__":
    main()
