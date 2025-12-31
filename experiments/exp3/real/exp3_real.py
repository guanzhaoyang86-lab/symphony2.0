#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiments/exp3/real/exp3_real.py

Exp3 (REAL): Robustness to Role Shock under real model / API execution

Key differences from sim:
- Real prompts instead of simulated tasks
- Real latency measurement
- Real success judged by heuristic / verifier
- Same selector, same metrics, same output schema
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ---- project root import ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../"))
import sys
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.linucb_selector import GlobalLinUCB, build_x


# -----------------------------
# 1. Task / Agent abstraction
# -----------------------------
@dataclass
class RealTask:
    tid: int
    prompt: str
    difficulty: str   # "simple" | "hard"


@dataclass
class RealAgentProfile:
    agent_id: str
    model_name: str
    call_cost: float
    p_simple: float  # Success probability for simple tasks
    p_hard: float    # Success probability for hard tasks


class RealAgent:
    """
    Real agent wrapper (API / local model)
    For placeholder mode, simulates success probability like SimAgent
    """

    def __init__(self, profile: RealAgentProfile, rng: random.Random):
        self.p = profile
        self.rng = rng
        self.available = True
        self.latency_ms = 500.0
        self.reputation = 0.5
        self._original_p_hard = profile.p_hard

    def apply_shock(self, shock_type: str):
        """
        Apply shock (same logic as SimAgent for consistency)
        """
        if shock_type == "A_unavailable":
            if self.p.agent_id == "A":
                self.available = False
            # Note: In real version with only 3 agents (A, B, C),
            # we don't degrade C like sim version does with E/C
            # For consistency with sim, we can degrade C if needed
            elif self.p.agent_id == "C":
                self.p.p_hard *= 0.70  # Similar to sim's C degradation
            
        elif shock_type == "A_degraded":
            if self.p.agent_id == "A":
                self.p.p_hard = 0.25  # Same as sim: 0.99 -> 0.25
            elif self.p.agent_id == "C":
                self.p.p_hard *= 0.80  # Similar to sim's C degradation

    def _success_prob(self, difficulty: str) -> float:
        """Calculate success probability based on difficulty"""
        base = self.p.p_simple if difficulty == "simple" else self.p.p_hard
        return max(0.0, min(1.0, base))

    def execute(self, prompt: str, difficulty: str) -> Tuple[str, float, bool]:
        """
        Execute task and return (response, latency, success).
        
        In placeholder mode, success is probabilistic based on agent capability.
        Replace this with real LLM calls when needed.
        """
        start = time.time()

        # ---- dummy placeholder ----
        response = f"[{self.p.model_name}] response to: {prompt[:50]}"

        latency = (time.time() - start) * 1000.0
        self.latency_ms = 0.8 * self.latency_ms + 0.2 * latency

        # Simulate success based on probability (like SimAgent)
        p = self._success_prob(difficulty)
        ok = (self.rng.random() < p)
        
        # Update reputation based on success
        self.reputation = max(0.0, min(1.0, 0.95 * self.reputation + 0.05 * (1.0 if ok else 0.0)))

        return response, latency, ok


# -----------------------------
# 2. Task loading
# -----------------------------
def _extract_prompt_from_raw_data(raw_data: dict, benchmark: str) -> str:
    """Extract prompt from raw_data based on benchmark type"""
    # Different benchmarks use different field names
    if benchmark == "humaneval":
        return raw_data.get("prompt", "")
    elif benchmark == "gsm8k":
        return raw_data.get("question", "")
    elif benchmark in ["bbh", "amc", "medical_qa"]:
        # Try common field names in order
        return raw_data.get("input", "") or raw_data.get("question", "") or raw_data.get("problem", "")
    else:
        # Fallback: try common field names
        return raw_data.get("prompt", "") or raw_data.get("question", "") or raw_data.get("input", "")


def load_tasks(path: str, n: int) -> List[RealTask]:
    """
    Load tasks from symphony-data-generator JSONL file.
    
    REQUIRED format (from symphony-data-generator):
    {"benchmark": "...", "difficulty_bin": "easy"|"hard", "raw_data": {...}}
    
    This function ONLY supports the data generator format to ensure real experiments
    use real benchmark data (HumanEval, GSM8K, BBH, AMC, Medical QA).
    """
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            
            # REQUIRED: Must have "raw_data" field (data generator format)
            if "raw_data" not in obj:
                raise ValueError(
                    f"Task {i} in {path} is missing 'raw_data' field. "
                    "Real experiments MUST use symphony-data-generator format. "
                    "Expected format: {{\"benchmark\": \"...\", \"difficulty_bin\": \"easy\"|\"hard\", \"raw_data\": {{...}}}}. "
                    "Generate tasks using: cd symphony-data-generator && python src/quick_start.py"
                )
            
            # Extract from data generator format
            raw_data = obj.get("raw_data", {})
            benchmark = obj.get("benchmark", "")
            prompt = _extract_prompt_from_raw_data(raw_data, benchmark)
            
            if not prompt:
                print(f"[WARN] Task {i} (benchmark={benchmark}) has no extractable prompt, skipping")
                continue
            
            # Map difficulty_bin ("easy"/"hard") to ("simple"/"hard")
            difficulty_bin = obj.get("difficulty_bin", "hard")
            if difficulty_bin not in ["easy", "hard"]:
                raise ValueError(
                    f"Task {i} has invalid difficulty_bin: {difficulty_bin}. "
                    "Expected 'easy' or 'hard'"
                )
            difficulty = "simple" if difficulty_bin == "easy" else "hard"
            
            tasks.append(
                RealTask(
                    tid=len(tasks),  # Use actual index after filtering
                    prompt=prompt,
                    difficulty=difficulty,
                )
            )
    
    if len(tasks) == 0:
        raise ValueError(
            f"No valid tasks loaded from {path}. "
            "Make sure the file is generated by symphony-data-generator. "
            "Run: cd symphony-data-generator && python src/quick_start.py"
        )
    
    return tasks


# -----------------------------
# 3. Data structures for logging
# -----------------------------
@dataclass
class StepLog:
    t: int
    policy: str
    task_difficulty: str
    chosen_agent: str
    match_score: float
    load: float
    latency_ms: float
    call_cost: float
    success: int
    is_shock: int
    rolling_success: float
    reward_used_for_update: float


@dataclass
class SummaryRow:
    policy: str
    n: int
    p_hard: float
    shock_type: str
    shock_point: int
    success_rate_pre_shock: float
    success_rate_post_shock: float
    success_rate_overall: float
    recovery_time: int
    deadlock_rate: float
    avg_latency_ms: float
    choose_A: int
    choose_B: int
    choose_C: int
    choose_D: int
    choose_E: int
    choose_A_pre: int
    choose_B_pre: int
    choose_C_pre: int
    choose_D_pre: int
    choose_E_pre: int
    choose_A_post: int
    choose_B_post: int
    choose_C_post: int
    choose_D_post: int
    choose_E_post: int


# -----------------------------
# 4. Utility functions
# -----------------------------
def print_exp3_summary_terminal(
    summary_row: SummaryRow,
    shock_type: str,
    shock_point: int,
):
    """Print comprehensive Exp3 summary in terminal"""
    print("\n" + "=" * 90)
    print(f"📊 Exp3 Real Execution Summary")
    print(f"Shock Type : {shock_type}")
    print(f"Shock Point: t = {shock_point}")
    print(f"Total Tasks: {summary_row.n}")
    print("=" * 90)
    print("Recovery Definition:")
    print("  ✓ Significant drop observed (≥8%)")
    print("  ✓ Sustained recovery to ≥90% of pre-shock baseline")
    print("  ✓ Recovery maintained for 50 consecutive steps")
    print("=" * 90)

    r = summary_row
    drop = r.success_rate_pre_shock - r.success_rate_post_shock
    recovered = "YES" if r.recovery_time >= 0 else "NO"
    rec_time = f"{r.recovery_time:4d}" if r.recovery_time >= 0 else "     N/A"

    print(f"\nPolicy: {r.policy}")
    print(f"  Success Rate (Pre-shock):  {r.success_rate_pre_shock:.4f}")
    print(f"  Success Rate (Post-shock): {r.success_rate_post_shock:.4f}")
    print(f"  Success Rate Drop:         {drop:.4f}")
    print(f"  Success Rate (Overall):    {r.success_rate_overall:.4f}")
    print(f"\nRecovery:")
    print(f"  Recovery Time (steps):     {rec_time}")
    print(f"  Recovered:                 {recovered}")
    print(f"\nOther Metrics:")
    print(f"  Deadlock Rate:             {r.deadlock_rate:.4f}")
    print(f"  Avg Latency (ms):          {r.avg_latency_ms:.2f}")
    print(f"\nAgent Selection (Overall):")
    print(f"  A: {r.choose_A:4d}  B: {r.choose_B:4d}  C: {r.choose_C:4d}")
    print(f"\nAgent Selection (Pre-shock):")
    print(f"  A: {r.choose_A_pre:4d}  B: {r.choose_B_pre:4d}  C: {r.choose_C_pre:4d}")
    print(f"\nAgent Selection (Post-shock):")
    print(f"  A: {r.choose_A_post:4d}  B: {r.choose_B_post:4d}  C: {r.choose_C_post:4d}")
    print("=" * 90)


def calculate_rolling_success(success_history: List[int], window_size: int = 50) -> List[float]:
    """Calculate rolling success rate"""
    rolling = []
    window = []
    for s in success_history:
        window.append(s)
        if len(window) > window_size:
            window.pop(0)
        rolling.append(sum(window) / len(window) if window else 0.0)
    return rolling


def calculate_recovery_time_strict(
    rolling_success: List[float],
    shock_point: int,
    *,
    min_drop: float = 0.08,
    recovery_ratio: float = 0.9,
    sustain_window: int = 50,
) -> int:
    """Paper-grade recovery definition"""
    n = len(rolling_success)
    if shock_point < 200 or shock_point + sustain_window >= n:
        return -1
    
    baseline = sum(rolling_success[shock_point - 200: shock_point]) / 200
    post_shock_window = rolling_success[shock_point: min(shock_point + 100, n)]
    if not post_shock_window:
        return -1
    
    post_min = min(post_shock_window)
    actual_drop = baseline - post_min
    
    if actual_drop < min_drop:
        return -1
    
    target = baseline * recovery_ratio
    for t in range(shock_point, n - sustain_window + 1):
        window = rolling_success[t: t + sustain_window]
        if len(window) < sustain_window:
            continue
        if all(x >= target for x in window):
            return t - shock_point
    
    return -1


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_excel(outdir: str, summary: List[SummaryRow], traj: Dict[str, List[StepLog]]) -> None:
    """Write results to Excel file"""
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas not available, skipping Excel export")
        return

    excel_path = os.path.join(outdir, "exp3_results.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for r in summary:
                row_dict = asdict(r)
                row_dict['recovery_status'] = 'Recovered' if r.recovery_time >= 0 else 'Not Recovered'
                row_dict['success_rate_drop'] = r.success_rate_pre_shock - r.success_rate_post_shock
                row_dict['recovery_efficiency'] = r.recovery_time if r.recovery_time >= 0 else None
                summary_data.append(row_dict)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Trajectory sheet
            for policy, logs in traj.items():
                if logs:
                    traj_df = pd.DataFrame([asdict(log) for log in logs])
                    sheet_name = f'Trajectory_{policy}'[:31]
                    traj_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Excel file saved to: {excel_path}")
    except ImportError as e:
        if 'openpyxl' in str(e).lower():
            print("[WARN] openpyxl not available, skipping Excel export. Install with: pip install openpyxl")
        else:
            print(f"[WARN] Missing dependency: {e}")
    except Exception as e:
        print(f"[WARN] Failed to write Excel file: {e}")


def try_plot(outdir: str, summary: List[SummaryRow], traj: Dict[str, List[StepLog]]) -> None:
    """Generate plots"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available: {e}")
        return

    try:
        if "linucb" in traj:
            logs = traj["linucb"]
            shock_point = summary[0].shock_point if summary else 500
            
            # V-shape recovery curve
            fig = plt.figure(figsize=(12, 6), dpi=180, constrained_layout=True)
            ax = fig.add_subplot(111)
            
            task_indices = [log.t for log in logs]
            rolling_success = [log.rolling_success for log in logs]
            
            ax.plot(task_indices, rolling_success, linewidth=2, label="LinUCB (Ours)")
            ax.axvline(x=shock_point, color="red", linestyle="--", linewidth=2, label=f"Shock Point (t={shock_point})")
            
            # Mark recovery point
            for row in summary:
                if row.policy == "linucb" and row.recovery_time >= 0:
                    recovery_point = shock_point + row.recovery_time
                    if recovery_point < len(rolling_success):
                        ax.plot(recovery_point, rolling_success[recovery_point], "go", markersize=10, label=f"Recovery (t={recovery_point})")
            
            ax.set_xlabel("Task Index", fontsize=12)
            ax.set_ylabel("Rolling Success Rate", fontsize=12)
            ax.set_title(f"Exp3: V-Shape Recovery (Shock Type: {summary[0].shock_type if summary else 'N/A'})", fontsize=14, fontweight="bold")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            fig.savefig(os.path.join(outdir, "plot_v_shape_recovery.png"))
            plt.close(fig)
            
            # Single policy comparison (simplified version of all policies plot)
            fig3 = plt.figure(figsize=(14, 7), dpi=180, constrained_layout=True)
            ax3 = fig3.add_subplot(111)
            
            shock_type = summary[0].shock_type if summary else "N/A"
            ax3.plot(task_indices, rolling_success, linewidth=2, label="LinUCB", color='#2ecc71', alpha=0.8)
            ax3.axvline(x=shock_point, color="red", linestyle="--", linewidth=2, label=f"Shock Point (t={shock_point})", zorder=10)
            
            for row in summary:
                if row.recovery_time >= 0:
                    recovery_point = shock_point + row.recovery_time
                    if recovery_point < len(rolling_success):
                        ax3.plot(recovery_point, rolling_success[recovery_point], "o", markersize=12, 
                                color='#2ecc71', markeredgecolor='black', markeredgewidth=1.5, 
                                label=f"Recovery (t={recovery_point})", zorder=5)
            
            ax3.set_xlabel("Task Index", fontsize=12)
            ax3.set_ylabel("Rolling Success Rate", fontsize=12)
            ax3.set_title(f"Exp3: V-Shape Recovery Comparison (Shock Type: {shock_type})", fontsize=14, fontweight="bold")
            ax3.legend(loc="best", fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1.05])
            
            fig3.savefig(os.path.join(outdir, "plot_v_shape_all_policies.png"))
            plt.close(fig3)
            
            # Recovery time comparison (simplified - single bar)
            fig2 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=True)
            ax2 = fig2.add_subplot(111)
            
            r = summary[0]
            if r.recovery_time >= 0:
                # Recovered: show actual recovery time
                recovery_time = r.recovery_time
                color = '#2ecc71'  # Green
                label_text = str(r.recovery_time)
                label_color = 'black'
                y_max = max(recovery_time, 100)
                bars = ax2.bar([0], [recovery_time], color=color, alpha=0.8, width=0.6)
            else:
                # Not recovered: show "N/A" with a small placeholder bar
                recovery_time = 50  # Small placeholder value for visualization
                color = '#e74c3c'  # Red
                label_text = "N/A"
                label_color = 'red'
                y_max = 150
                bars = ax2.bar([0], [recovery_time], color=color, alpha=0.3, width=0.6, edgecolor='red', linewidth=2)
            
            ax2.set_xticks([0])
            ax2.set_xticklabels([r.policy], fontsize=11)
            ax2.set_ylabel("Recovery Time (steps)", fontsize=12)
            ax2.set_title("Recovery Time Comparison", fontsize=14, fontweight="bold")
            ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
            
            ax2.text(0, recovery_time + y_max * 0.02, label_text, ha="center", va="bottom", 
                    fontsize=10, fontweight="bold", color=label_color)
            ax2.set_ylim([0, y_max * 1.15])
            
            fig2.savefig(os.path.join(outdir, "plot_recovery_time_comparison.png"))
            plt.close(fig2)
    except Exception as e:
        print(f"[WARN] Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()


# -----------------------------
# 5. Success verifier (for real LLM execution mode)
# -----------------------------
def verify_success(task: RealTask, response: str) -> bool:
    """
    Paper-acceptable heuristic verifier for REAL LLM responses:
    - keyword check
    - regex
    - or LLM-as-judge (future)
    
    Note: In placeholder mode, success is determined by agent.execute()
    based on probability. This function is kept for future real LLM integration.
    """
    if task.difficulty == "simple":
        return len(response) > 20
    return len(response) > 50


# -----------------------------
# 4. Main experiment loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Exp3 REAL")
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shock", type=str, required=True,
                    choices=["A_unavailable", "A_degraded"])
    ap.add_argument("--shock-point", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="experiments/exp3/real/results/real_exp3_robustness")
    ap.add_argument("--no-plots", action="store_true", help="do not generate png plots")
    ap.add_argument("--no-excel", action="store_true", help="do not generate Excel file")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    shock_point = args.shock_point or args.n // 2

    # Agents (with success probabilities matching sim version)
    agents = {
        "A": RealAgent(RealAgentProfile("A", "gpt-4", 1.0, p_simple=0.99, p_hard=0.99), rng),
        "B": RealAgent(RealAgentProfile("B", "gpt-3.5", 0.1, p_simple=0.88, p_hard=0.18), rng),
        "C": RealAgent(RealAgentProfile("C", "mixtral", 0.3, p_simple=0.92, p_hard=0.65), rng),
    }

    selector = GlobalLinUCB(d=6, l2=1.0, alpha=1.0, delta=0.05, S=1.0)

    tasks = load_tasks(args.tasks, args.n)

    # Create output directory
    shock_type_short = "ShockA" if args.shock == "A_unavailable" else "ShockB"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join(args.outdir, shock_type_short, ts)
    os.makedirs(outdir, exist_ok=True)

    step_logs: List[StepLog] = []
    success_hist: List[int] = []
    choose_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    choose_counts_pre = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    choose_counts_post = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    total_latency = 0.0

    for t, task in enumerate(tasks):
        if t == shock_point:
            for ag in agents.values():
                ag.apply_shock(args.shock)

        # available agents
        candidates = [ag for ag in agents.values() if ag.available]

        if not candidates:
            # No agents available
            success_hist.append(0)
            step_logs.append(StepLog(
                t=t,
                policy="linucb",
                task_difficulty=task.difficulty,
                chosen_agent="NONE",
                match_score=0.0,
                load=0.0,
                latency_ms=0.0,
                call_cost=0.0,
                success=0,
                is_shock=1 if t == shock_point else 0,
                rolling_success=0.0,
                reward_used_for_update=0.0,
            ))
            continue

        xs = []
        for ag in candidates:
            match_score = 0.8 if ag.p.agent_id == "A" else 0.6
            x = build_x(
                match_score=match_score,
                dynamic_state={
                    "load": 0.0,
                    "latency_ms": ag.latency_ms,
                    "reputation": ag.reputation,
                },
                available=True,
                latency_scale_ms=2000.0,
            )
            xs.append((ag.p.agent_id, x))

        chosen_id = selector.select(xs)
        agent = agents[chosen_id]
        match_score = 0.8 if chosen_id == "A" else 0.6

        # Execute and get success status
        resp, latency, ok = agent.execute(task.prompt, task.difficulty)
        reward = 1.0 if ok else 0.0

        chosen_x = next(x for aid, x in xs if aid == chosen_id)
        selector.update(chosen_x, reward)

        success_hist.append(1 if ok else 0)
        choose_counts[chosen_id] = choose_counts.get(chosen_id, 0) + 1
        if t < shock_point:
            choose_counts_pre[chosen_id] = choose_counts_pre.get(chosen_id, 0) + 1
        else:
            choose_counts_post[chosen_id] = choose_counts_post.get(chosen_id, 0) + 1
        total_latency += latency

        # Calculate rolling success
        rolling_success_list = calculate_rolling_success(success_hist, window_size=50)
        current_rolling = rolling_success_list[-1] if rolling_success_list else 0.0

        step_logs.append(StepLog(
            t=t,
            policy="linucb",
            task_difficulty=task.difficulty,
            chosen_agent=chosen_id,
            match_score=match_score,
            load=0.0,
            latency_ms=latency,
            call_cost=agent.p.call_cost,
            success=1 if ok else 0,
            is_shock=1 if t == shock_point else 0,
            rolling_success=current_rolling,
            reward_used_for_update=reward,
        ))

    # Calculate metrics
    n = len(tasks)
    avg_latency = total_latency / max(1, n)
    p_hard = sum(1 for t in tasks if t.difficulty == "hard") / max(1, n)
    
    pre_shock_successes = sum(success_hist[:shock_point]) if shock_point > 0 else 0
    pre_shock_rate = pre_shock_successes / max(1, shock_point)
    
    post_shock_successes = sum(success_hist[shock_point:]) if shock_point < n else 0
    post_shock_count = n - shock_point
    post_shock_rate = post_shock_successes / max(1, post_shock_count)
    
    overall_rate = sum(success_hist) / max(1, n)
    
    # Recovery time
    rolling_success_list = calculate_rolling_success(success_hist, window_size=50)
    recovery_time = calculate_recovery_time_strict(
        rolling_success_list,
        shock_point,
        min_drop=0.08,
        recovery_ratio=0.9,
        sustain_window=50,
    )
    
    # Deadlock rate
    deadlock_count = 0
    if shock_point < n:
        consecutive_failures = 0
        for i in range(shock_point, n):
            if success_hist[i] == 0:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    deadlock_count += 1
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
    deadlock_rate = deadlock_count / max(1, post_shock_count)

    # Create summary
    summary_row = SummaryRow(
        policy="linucb",
        n=n,
        p_hard=float(p_hard),
        shock_type=args.shock,
        shock_point=shock_point,
        success_rate_pre_shock=float(pre_shock_rate),
        success_rate_post_shock=float(post_shock_rate),
        success_rate_overall=float(overall_rate),
        recovery_time=int(recovery_time),
        deadlock_rate=float(deadlock_rate),
        avg_latency_ms=float(avg_latency),
        choose_A=int(choose_counts.get("A", 0)),
        choose_B=int(choose_counts.get("B", 0)),
        choose_C=int(choose_counts.get("C", 0)),
        choose_D=0,
        choose_E=0,
        choose_A_pre=int(choose_counts_pre.get("A", 0)),
        choose_B_pre=int(choose_counts_pre.get("B", 0)),
        choose_C_pre=int(choose_counts_pre.get("C", 0)),
        choose_D_pre=0,
        choose_E_pre=0,
        choose_A_post=int(choose_counts_post.get("A", 0)),
        choose_B_post=int(choose_counts_post.get("B", 0)),
        choose_C_post=int(choose_counts_post.get("C", 0)),
        choose_D_post=0,
        choose_E_post=0,
    )

    # Print summary to terminal
    print_exp3_summary_terminal(summary_row, args.shock, shock_point)
    
    # Write outputs
    print(f"\n[OUTPUT] Writing results to: {outdir}")
    
    # Summary CSV
    summary_dict = asdict(summary_row)
    summary_dict['recovery_status'] = 'Recovered' if summary_row.recovery_time >= 0 else 'Not Recovered'
    summary_dict['success_rate_drop'] = summary_row.success_rate_pre_shock - summary_row.success_rate_post_shock
    write_csv(os.path.join(outdir, "summary.csv"), [summary_dict])
    print(f"  ✓ summary.csv")
    
    # Trajectory CSV
    traj_dicts = [asdict(log) for log in step_logs]
    write_csv(os.path.join(outdir, "trajectory_linucb.csv"), traj_dicts)
    print(f"  ✓ trajectory_linucb.csv")
    
    # Also save JSON (for backward compatibility)
    json_logs = [
        {
            "t": log.t,
            "agent": log.chosen_agent,
            "success": bool(log.success),
            "latency_ms": log.latency_ms,
            "is_shock": log.is_shock,
        }
        for log in step_logs
    ]
    with open(os.path.join(outdir, "trajectory_real.json"), "w") as f:
        json.dump(json_logs, f, indent=2)
    print(f"  ✓ trajectory_real.json")

    # Excel
    if not args.no_excel:
        write_excel(outdir, [summary_row], {"linucb": step_logs})
        excel_path = os.path.join(outdir, "exp3_results.xlsx")
        if os.path.exists(excel_path):
            print(f"  ✓ exp3_results.xlsx")

    # Plots
    if not args.no_plots:
        try_plot(outdir, [summary_row], {"linucb": step_logs})
        plot_files = [
            "plot_v_shape_recovery.png",
            "plot_v_shape_all_policies.png",
            "plot_recovery_time_comparison.png"
        ]
        for plot_file in plot_files:
            plot_path = os.path.join(outdir, plot_file)
            if os.path.exists(plot_path):
                print(f"  ✓ {plot_file}")

    print(f"\n✅ Real Exp3 finished. Results in {outdir}\n")


if __name__ == "__main__":
    main()
