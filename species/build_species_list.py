#!/usr/bin/env python3
"""build_species_list.py — Aurora-500: the mainstream North American species list.

Derives the list from iNaturalist observation PREVALENCE (research-grade, wild
observations, United States + Canada merged), per taxon group:

    Aves       250   Mammalia    120   Reptilia     70   Amphibia     40
                                                  + 20 exam species  (kept)
    -> Aurora-500 (JSON with common name, scientific name, group, count, danger)

Why 500 and not 1000: the embedding table costs 1.5 KB/species (768 fp16), so
size was never the constraint — encounter probability is. Species 500-1000 in NA
prevalence are mostly rarely-photographed rodents/bats/shrews/vagrants: they add
lookalike confusion and curation burden with near-zero user value. The pipeline
is count-agnostic, so growing to 1000/2000 later is a re-run of this script with
larger quotas (append-only embedding table).

  python build_species_list.py            # -> aurora-500.json (+ build report)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
API = "https://api.inaturalist.org/v1"
QUOTAS = {"Aves": 250, "Mammalia": 120, "Reptilia": 70, "Amphibia": 40}
GROUPS = list(QUOTAS)
# Sealed-exam species from animals.json — always present so the original exam set stays valid.
EXAM_20 = [
    ("white-tailed deer", "Odocoileus virginianus"), ("mule deer", "Odocoileus hemionus"),
    ("eastern gray squirrel", "Sciurus carolinensis"), ("eastern chipmunk", "Tamias striatus"),
    ("eastern cottontail rabbit", "Sylvilagus floridanus"), ("raccoon", "Procyon lotor"),
    ("red fox", "Vulpes vulpes"), ("coyote", "Canis latrans"),
    ("striped skunk", "Mephitis mephitis"), ("virginia opossum", "Didelphis virginiana"),
    ("black bear", "Ursus americanus"), ("grizzly bear", "Ursus arctos"),
    ("american bison", "Bos bison"), ("pronghorn", "Antilocapra americana"),
    ("bald eagle", "Haliaeetus leucocephalus"), ("great blue heron", "Ardea herodias"),
    ("canada goose", "Branta canadensis"), ("wild turkey", "Meleagris gallopavo"),
    ("mallard", "Anas platyrhynchos"), ("american crow", "Corvus brachyrhynchos"),
]


def get(url: str, params: dict, retries: int = 3) -> dict:
    q = urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "aurora-prevalence/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001 — retry network hiccups
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} ({exc.__class__.__name__})")
            time.sleep(3 * (attempt + 1))
    return {}


def place_id(name: str) -> int:
    data = get(f"{API}/places/autocomplete", {"q": name})
    for p in data.get("results", []):
        if p["name"].lower() == name.lower():
            return p["id"]
    sys.exit(f"iNat place not found: {name}")


def top_species(pid: int, group: str) -> list[dict]:
    """Ranked species_counts for one place + group (research-grade, wild)."""
    data = get(f"{API}/observations/species_counts", {
        "place_id": pid, "iconic_taxa": group, "quality_grade": "research",
        "captive": "false", "per_page": 500,
    })
    out = []
    for row in data.get("results", []):
        t = row.get("taxon") or {}
        if t.get("rank") != "species" or not t.get("name"):
            continue
        out.append({"sci": t["name"], "name": (t.get("preferred_common_name") or "").lower()
                    or t["name"].lower(), "count": int(row["count"])})
    return out


def main() -> None:
    us, ca = place_id("United States"), place_id("Canada")
    print(f"[species] places: US={us} Canada={ca}")
    merged: dict[str, dict] = {}
    for group in GROUPS:
        per_place = [top_species(pid, group) for pid in (us, ca)]
        time.sleep(1.0)
        agg: dict[str, dict] = {}
        for place in per_place:  # sum US+Canada counts = NA prevalence proxy
            for s in place:
                a = agg.setdefault(s["sci"], {"sci": s["sci"], "name": s["name"], "count": 0})
                a["count"] += s["count"]
        ranked = sorted(agg.values(), key=lambda x: -x["count"])[: QUOTAS[group]]
        for rank, s in enumerate(ranked, 1):
            s.update(group=group, rank=rank)
            merged[s["sci"]] = s
        sample = ", ".join(f"{s['name']} ({s['sci']})" for s in ranked[:3])
        print(f"[species] {group:<10} top-{QUOTAS[group]} merged (kept {len(ranked)}): {sample}")

    kept = set(merged)
    added_exam = []
    for name, sci in EXAM_20:
        if sci not in kept:  # exam anchors must never fall out of the list
            merged[sci] = {"sci": sci, "name": name, "count": 0, "group": "exam-anchor", "rank": 0}
            added_exam.append(name)
    if added_exam:
        print(f"[species] re-added exam anchors: {', '.join(added_exam)}")

    overlay = json.loads((ROOT / "danger_overlay.json").read_text(encoding="utf-8"))
    for sci, d in overlay.items():
        if sci in merged:
            merged[sci].update(danger=d["danger"], danger_note=d["note"])
        else:  # safety-relevant species must ship even if not in the top-N
            merged[sci] = {"sci": sci, "name": d["name"], "count": 0,
                           "group": d.get("group", "safety"), "rank": 0,
                           "danger": d["danger"], "danger_note": d["note"]}
    n_danger = sum(1 for s in merged.values() if s.get("danger") in ("high", "medium"))

    species = sorted(merged.values(), key=lambda s: (s["group"], s["rank"], -s["count"]))
    out = ROOT / "aurora-500.json"
    out.write_text(json.dumps({
        "name": "Aurora-500", "generated_by": "build_species_list.py",
        "source": "iNaturalist species_counts (US+Canada, research-grade, captive=false)",
        "quotas": QUOTAS, "species": species,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    groups = {}
    for s in species:
        groups[s["group"]] = groups.get(s["group"], 0) + 1
    print(f"[species] Aurora-500: {len(species)} species {groups} | "
          f"danger-flagged: {n_danger} -> {out.name}")


if __name__ == "__main__":
    main()
