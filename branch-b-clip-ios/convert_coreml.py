#!/usr/bin/env python3
"""convert_coreml.py — BioCLIP-2 image encoder -> Core ML mlpackage (fp16).

Produces the artifact that runs natively on the iPhone 13 Neural Engine:
only the ViT-L/14 IMAGE encoder (the text tower never ships — see README.md).

Run on Linux (WSL2) or macOS. Prediction-time validation requires macOS
(coremltools predict is macOS-only); use validate_coreml.py there, then the
on-device exam on the iPhone 13 itself.

  python convert_coreml.py       # -> coreml/BioCLIP2-ImageEncoder.mlpackage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: PLC0415

MODEL = "hf-hub:imageomics/bioclip-2"
OUT = Path(__file__).parent / "coreml" / "BioCLIP2-ImageEncoder.mlpackage"
# CLIP normalization constants (identical to open_clip preprocess for bioclip-2)
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def load_model():
    import math  # noqa: PLC0415
    import open_clip  # noqa: PLC0415
    # coremltools cannot convert torch's fused attention paths — disable the
    # nn.MultiheadAttention C++ fastpath and decompose scaled_dot_product_attention
    # into plain matmul/softmax before tracing.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:  # noqa: BLE001 — older torch without the switch
        pass
    import torch.nn.functional as F  # noqa: PLC0415

    def _sdpa_math(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
        scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        if is_causal:
            size = query.size(-2)
            attn_weight = attn_weight + torch.triu(
                torch.ones(size, size, dtype=query.dtype) * float("-inf"), diagonal=1)
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        return torch.softmax(attn_weight, dim=-1) @ value

    F.scaled_dot_product_attention = _sdpa_math
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


def main() -> None:
    import coremltools as ct  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    model, preprocess = load_model()
    visual = model.visual.eval()  # returns the projected 768-d embedding
    example = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(visual, example)

    print("[coreml] tracing ok — converting (fp16, mlprogram)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, 224, 224), dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        outputs=[ct.TensorType(name="embedding")],
    )
    mlmodel.author = "privatelens species-id package"
    mlmodel.license = "BioCLIP-2 weights: MIT (imageomics/bioclip-2)"
    mlmodel.short_description = ("BioCLIP-2 ViT-L/14 image encoder, fp16. "
                                 "Returns L2-normalizable 768-d embedding; "
                                 "cosine vs species_embeddings.f16.bin = zero-shot species logits.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6

    spec = json.loads(json.dumps({"mean": MEAN, "std": STD, "input": [1, 3, 224, 224],
                                  "output": [1, 768], "model": MODEL}))
    (OUT.parent / "coreml_meta.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"[coreml] -> {OUT} ({size:.0f} MB)\n"
          f"[coreml] preprocess (bake into Swift): resize 224 bicubic-antialias (shortest side), "
          f"center crop, /255, (x-mean)/std\n"
          f"next (on a Mac): python validate_coreml.py --images <some photos>")


if __name__ == "__main__":
    sys.exit(main())
