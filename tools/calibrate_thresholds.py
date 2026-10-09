#!/usr/bin/env python3
"""
Threshold Calibration & Similarity Distribution Analyzer
=========================================================
Reads recorded session JSONL logs, analyzes cosine similarity distributions
between same-identity samples vs distinct-identity samples, and calculates
optimal data-driven thresholds for:
  - MATCH_SIM_THRESHOLD (cross-camera assignment)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np


def analyze_log(log_path: Path):
    print(f"\n[Calibration] Loading session log: {log_path.name}")
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
        return

    embeddings = []
    labels = []
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            evt = entry.get("event")
            data = entry.get("data", {})
            if evt == "TRACKLET_EMBEDDING" and "embedding" in data:
                embeddings.append(np.array(data["embedding"], dtype=np.float32))
                labels.append(data.get("assigned_gid", "UNKNOWN"))

    if len(embeddings) < 10:
        print(f"Insufficient embeddings found ({len(embeddings)}). Need at least 10 samples.")
        return

    embeddings = np.stack(embeddings)
    # Pairwise cosine similarity matrix
    sim_mat = embeddings @ embeddings.T

    same_sims = []
    diff_sims = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == "UNKNOWN" or labels[j] == "UNKNOWN":
                continue
            sim = float(sim_mat[i, j])
            if labels[i] == labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    print("\n" + "=" * 60)
    print("  SIMILARITY DISTRIBUTION ANALYSIS")
    print("=" * 60)
    if same_sims:
        print(f"Same-Person Pairs     (N={len(same_sims)}):")
        print(f"  Mean: {np.mean(same_sims):.3f} | Median: {np.median(same_sims):.3f}")
        print(f"  Min:  {np.min(same_sims):.3f} | Max:    {np.max(same_sims):.3f}")
        print(f"  10th Percentile: {np.percentile(same_sims, 10):.3f}")
    if diff_sims:
        print(f"Different-Person Pairs (N={len(diff_sims)}):")
        print(f"  Mean: {np.mean(diff_sims):.3f} | Median: {np.median(diff_sims):.3f}")
        print(f"  Min:  {np.min(diff_sims):.3f} | Max:    {np.max(diff_sims):.3f}")
        print(f"  95th Percentile: {np.percentile(diff_sims, 95):.3f}")

    if same_sims and diff_sims:
        suggested_match = float(np.percentile(same_sims, 15))
        print("-" * 60)
        print("  RECOMMENDED DATA-DRIVEN THRESHOLDS:")
        print(f"  MATCH_SIM_THRESHOLD: {suggested_match:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Re-ID thresholds from session log")
    parser.add_argument("log", type=str, help="Path to session JSONL log file")
    args = parser.parse_args()
    analyze_log(Path(args.log))
