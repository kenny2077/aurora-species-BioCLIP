#!/usr/bin/env python3
"""make_manifest.py — sha256 manifest for each species-id package.

Writes manifest.json inside branch-a-survival/ and species-classifier/ covering
every file (streamed hashing; the multi-GB weights included). Aurora's
Ed25519 + SHA-256 activation flow signs this manifest as the trust root —
here it is generated unsigned; the signing step plugs in at catalog time.

  python make_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
PACKAGES = ["branch-a-survival", "species-classifier"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for pkg in PACKAGES:
        root = ROOT / pkg
        files = {}
        t0 = time.time()
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                rel = p.relative_to(root).as_posix()
                if p.suffix in {".exe", ".dll"} and "bin/" in rel:
                    continue  # third-party runtime binaries: pin by upstream release tag instead
                files[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        manifest = {
            "package": pkg,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_count": len(files),
            "total_bytes": sum(f["bytes"] for f in files.values()),
            "llama_cpp_binaries": "llama.cpp b10819 win-cuda-12.4-x64 (pinned upstream release)",
            "files": files,
        }
        out = root / "manifest.json"
        out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"[manifest] {pkg}: {len(files)} files, "
              f"{manifest['total_bytes']/1e9:.2f} GB, {time.time()-t0:.0f}s -> manifest.json")


if __name__ == "__main__":
    main()
