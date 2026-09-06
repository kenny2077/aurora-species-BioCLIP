#!/usr/bin/env python3
"""fetch_exam_photos.py — expand the sealed exam for the iPhone-13 run.

The original sealed set (data/test/, 39 photos / 20 species) stays untouched.
This script adds 2 fresh CC-licensed iNaturalist photos for each of 24 more
Aurora-500 species (mammals, birds, venomous reptiles, amphibians) into
photos/<species>/ and maintains photo_labels.json (folder -> scientific name)
plus a per-exam provenance.csv. Resumable: species with >=2 photos are skipped.

  python fetch_exam_photos.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
PHOTOS = ROOT / "photos"
SPECIES_FILE = ROOT.parent / "species" / "aurora-500.json"
INAT = "https://api.inaturalist.org/v1/observations"
SHIP_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
PER_SPECIES = 2

# 24 additional species: 6 mammals, 8 birds, 5 reptiles (3 venomous), 5 amphibians.
EXAM24_SCI = [
    "Alces alces", "Cervus canadensis", "Lynx rufus", "Canis lupus",
    "Castor canadensis", "Lontra canadensis",
    "Turdus migratorius", "Cardinalis cardinalis", "Cyanocitta cristata",
    "Buteo jamaicensis", "Cathartes aura", "Dryobates pubescens",
    "Bubo virginianus", "Spinus tristis",
    "Agkistrodon contortrix", "Crotalus horridus", "Crotalus atrox",
    "Alligator mississippiensis", "Terrapene carolina",
    "Lithobates catesbeianus", "Anaxyrus americanus",
    "Notophthalmus viridescens", "Pseudacris crucifer",
]


def load_list() -> dict[str, dict]:
    cfg = json.loads(SPECIES_FILE.read_text(encoding="utf-8"))
    return {s["sci"]: s for s in cfg["species"]}


def candidates(session, sci: str) -> list[dict]:
    """CC-licensed research-grade photo candidates, taxon-verified (same rule as lora.py)."""
    seen, cands = set(), []
    for order in ("votes", "random"):
        q = urllib.parse.urlencode({
            "taxon_name": sci, "photos": "true", "quality_grade": "research",
            "photo_license": ",".join(sorted(SHIP_LICENSES)),
            "per_page": 12, "order_by": order,
        })
        req = urllib.request.Request(f"{INAT}?{q}",
                                     headers={"User-Agent": "aurora-exam/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            for obs in json.load(r).get("results", []):
                taxon = ((obs.get("taxon") or {}).get("name") or "").lower()
                if not taxon.startswith(sci.lower()):
                    continue
                photo = obs["photos"][0]
                code = (photo.get("license_code") or "").lower()
                if code not in SHIP_LICENSES or photo["id"] in seen:
                    continue
                seen.add(photo["id"])
                cands.append({"url": photo["url"].replace("square", "medium"),
                              "license": code, "attribution": obs.get("attribution", "")})
        time.sleep(1.0)
    return cands


def main() -> None:
    import requests  # noqa: PLC0415
    by_sci = load_list()
    missing = [s for s in EXAM24_SCI if s not in by_sci]
    if missing:
        sys.exit(f"species not in Aurora-500 (fix list first): {missing}")

    labels_path = ROOT / "photo_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    prov_path = ROOT / "provenance.csv"
    new_prov = not prov_path.exists()
    session = requests.Session()
    session.headers.update({"User-Agent": "aurora-exam/1.0"})

    with prov_path.open("a", newline="", encoding="utf-8") as prov:
        writer = csv.writer(prov)
        if new_prov:
            writer.writerow(["species", "sci", "url", "license", "attribution", "file"])
        for sci in EXAM24_SCI:
            name = by_sci[sci]["name"]
            folder = PHOTOS / name
            have = len(list(folder.glob("*"))) if folder.exists() else 0
            if have >= PER_SPECIES:
                labels[name] = sci
                print(f"[fetch] {name}: have {have} — skipped")
                continue
            try:
                cands = candidates(session, sci)
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] {name}: API error {exc.__class__.__name__} — retry later")
                continue
            random.Random(13).shuffle(cands)
            got = 0
            for c in cands:
                if got >= PER_SPECIES - have:
                    break
                photo_id = c["url"].split("/photos/")[1].split("/")[0]
                dst = folder / f"{photo_id}.jpg"
                try:
                    with urllib.request.urlopen(urllib.request.Request(
                            c["url"], headers={"User-Agent": "aurora-exam/1.0"}), timeout=30) as r:
                        data = r.read()
                    if len(data) < 10000:
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                except Exception:  # noqa: BLE001
                    continue
                writer.writerow([name, sci, c["url"], c["license"], c["attribution"], str(dst)])
                got += 1
                time.sleep(0.5)
            labels[name] = sci
            print(f"[fetch] {name:<36} +{got} photos (had {have}, candidates {len(cands)})")
            time.sleep(1.0)

    labels_path.write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    total = sum(len(list(d.glob('*'))) for d in PHOTOS.iterdir() if d.is_dir())
    print(f"[fetch] exam pool: {len(labels)} folders, {total} photos | labels -> photo_labels.json")


if __name__ == "__main__":
    main()
