#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Exp3: Simulation vs Real execution (same figure)

Input:
- sim trajectory CSV (linucb)
- real trajectory JSON
"""

import argparse
import os
import json
import csv
from typing import List
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def rolling_success(xs: List[int], window: int = 50) -> List[float]:
    out = []
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


# -----------------------------
# Loaders
# -----------------------------
def load_sim_trajectory(path: str):
    ts, rs = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(int(row["t"]))
            rs.append(float(row["rolling_success"]))
    return ts, rs


def load_real_trajectory(path: str):
    ts, succ = [], []
    with open(path) as f:
        data = json.load(f)
        for row in data:
            ts.append(int(row["t"]))
            succ.append(1 if row["success"] else 0)
    rs = rolling_success(succ)
    return ts, rs


# -----------------------------
# Plot
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Plot Exp3: sim vs real")
    ap.add_argument("--sim", required=True, help="trajectory_linucb.csv")
    ap.add_argument("--real", required=True, help="trajectory_real.json")
    ap.add_argument("--shock-point-sim", type=int, help="Shock point for sim (if different from real)")
    ap.add_argument("--shock-point-real", type=int, help="Shock point for real (if different from sim)")
    ap.add_argument("--shock-point", type=int, help="Shock point (if same for both)")
    ap.add_argument("--shock-type", type=str, help="Shock type (A_unavailable or A_degraded) for title")
    ap.add_argument("--out", default="sim_vs_real.png")
    args = ap.parse_args()

    ts_sim, rs_sim = load_sim_trajectory(args.sim)
    ts_real, rs_real = load_real_trajectory(args.real)

    # Determine shock points
    shock_sim = args.shock_point_sim or args.shock_point
    shock_real = args.shock_point_real or args.shock_point
    
    if shock_sim is None or shock_real is None:
        raise ValueError("Must specify --shock-point or both --shock-point-sim and --shock-point-real")

    plt.figure(figsize=(14, 7), dpi=180)
    
    # Plot trajectories
    plt.plot(ts_sim, rs_sim, label="Simulation (LinUCB)", linewidth=2.5, 
             color='#3498db', alpha=0.9, marker='o', markersize=3, markevery=max(1, len(ts_sim)//50))
    plt.plot(ts_real, rs_real, label="Real Execution (LinUCB)", linewidth=2.5, 
             color='#e74c3c', alpha=0.9, marker='s', markersize=3, markevery=max(1, len(ts_real)//50))

    # Plot shock points
    if shock_sim <= max(ts_sim):
        plt.axvline(
            x=shock_sim,
            color="#3498db",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Sim Shock (t={shock_sim})",
        )
    
    if shock_real <= max(ts_real):
        plt.axvline(
            x=shock_real,
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Real Shock (t={shock_real})",
        )

    # Title
    shock_type_str = f" ({args.shock_type})" if args.shock_type else ""
    plt.xlabel("Task Index", fontsize=13, fontweight='bold')
    plt.ylabel("Rolling Success Rate", fontsize=13, fontweight='bold')
    plt.title(f"Exp3: Robustness under Role Shock{shock_type_str} (Sim vs Real)", 
              fontsize=15, fontweight='bold', pad=15)

    plt.ylim(0.0, 1.05)
    plt.xlim(0, max(max(ts_sim), max(ts_real)) * 1.02)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Add text annotation with key differences
    max_t = max(max(ts_sim), max(ts_real))
    plt.text(0.02, 0.98, 
             f"Sim: n={len(ts_sim)}, shock@t={shock_sim}\nReal: n={len(ts_real)}, shock@t={shock_real}",
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure saved to: {args.out}")
    print(f"   Sim: {len(ts_sim)} tasks, shock at t={shock_sim}")
    print(f"   Real: {len(ts_real)} tasks, shock at t={shock_real}")


if __name__ == "__main__":
    main()
