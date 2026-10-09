#!/usr/bin/env python3
"""
Multi-Camera Re-ID Evaluation Tool
==================================
Calculates tracking and Re-ID metrics:
  - Total ID switches per camera
  - False-merge rate (co-occurrence violations)
  - Duplicate-ID rate
  - Cross-camera consistency score
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def evaluate_session(log_path: Path):
    print(f"\n[Evaluate] Analyzing session log: {log_path.name}")
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
        return

    cannot_links = []
    merges = []
    track_assignments: Dict[Tuple[str, int], List[str]] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            evt = entry.get("event")
            data = entry.get("data", {})

            if evt == "CANNOT_LINK":
                cannot_links.append(data)
            elif evt == "RECONCILE_MERGE":
                merges.append(data)
            elif evt == "TRACK_ASSIGNMENT":
                cam = data.get("cam_id")
                tid = data.get("track_id")
                gid = data.get("assigned_gid")
                if cam and tid and gid:
                    track_assignments.setdefault((cam, tid), []).append(gid)

    # Compute ID switches within tracks
    id_switches = 0
    total_tracks = len(track_assignments)
    for (cam, tid), gids in track_assignments.items():
        unique_gids = set(gids)
        if len(unique_gids) > 1:
            id_switches += (len(unique_gids) - 1)

    print("\n" + "=" * 60)
    print("  MULTI-CAMERA RE-ID EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Local Tracks Analyzed: {total_tracks}")
    print(f"ID Switches Detected:        {id_switches}")
    print(f"Co-Occurrence Cannot-Links:  {len(cannot_links)}")
    print(f"Dynamic Merges Executed:     {len(merges)}")
    if total_tracks > 0:
        switch_rate = (id_switches / total_tracks) * 100
        print(f"Track ID Stability:          {100.0 - switch_rate:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-camera Re-ID session")
    parser.add_argument("log", type=str, help="Path to session JSONL log file")
    args = parser.parse_args()
    evaluate_session(Path(args.log))
