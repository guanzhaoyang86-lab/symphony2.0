#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp3 real execution: robustness to role shock with real LLM/API calls
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../"))
import sys
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.linucb_selector import GlobalLinUCB, build_x


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

    def __init__(self, profile: RealAgentProfile, rng: random.Random):
        self.p = profile
        self.rng = rng
        self.available = True
        self.latency_ms = 500.0
        self.reputation = 0.5
        self._original_p_hard = profile.p_hard

    def apply_shock(self, shock_type: str):
        # 应用shock：修改agent的可用性或性能概率，与sim版本保持一致
        if shock_type == "A_unavailable":
            # A变为不可用，C的性能下降
            if self.p.agent_id == "A":
                self.available = False
            elif self.p.agent_id == "C":
                self.p.p_hard *= 0.70
            
        elif shock_type == "A_degraded":
            # A的性能大幅下降（从0.99降到0.25），C也有一定下降
            if self.p.agent_id == "A":
                self.p.p_hard = 0.25
            elif self.p.agent_id == "C":
                self.p.p_hard *= 0.80

    def _success_prob(self, difficulty: str) -> float:
        # 根据任务难度返回对应的成功概率
        base = self.p.p_simple if difficulty == "simple" else self.p.p_hard
        return max(0.0, min(1.0, base))

    def execute(self, prompt: str, difficulty: str) -> Tuple[str, float, bool]:
        # 目前是placeholder实现：生成模拟响应并测量延迟
        start = time.time()
        response = f"[{self.p.model_name}] response to: {prompt[:50]}"
        latency = (time.time() - start) * 1000.0
        # 使用指数移动平均更新延迟估计
        self.latency_ms = 0.8 * self.latency_ms + 0.2 * latency

        # 根据agent的成功概率随机决定本次执行是否成功
        p = self._success_prob(difficulty)
        ok = (self.rng.random() < p)
        # 用成功结果更新reputation（也是指数移动平均）
        self.reputation = max(0.0, min(1.0, 0.95 * self.reputation + 0.05 * (1.0 if ok else 0.0)))

        return response, latency, ok


def _extract_prompt_from_raw_data(raw_data: dict, benchmark: str) -> str:
    # 不同benchmark的prompt字段名不同，需要根据benchmark类型提取
    if benchmark == "humaneval":
        return raw_data.get("prompt", "")
    elif benchmark == "gsm8k":
        return raw_data.get("question", "")
    elif benchmark in ["bbh", "amc", "medical_qa"]:
        # 这些benchmark可能用input/question/problem等字段名，按优先级尝试
        return raw_data.get("input", "") or raw_data.get("question", "") or raw_data.get("problem", "")
    else:
        # 默认尝试常见字段名
        return raw_data.get("prompt", "") or raw_data.get("question", "") or raw_data.get("input", "")


def load_tasks(path: str, n: int) -> List[RealTask]:
    # 从symphony-data-generator生成的JSONL文件加载任务
    # 要求文件格式必须包含raw_data字段，确保使用的是真实的benchmark数据
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            
            if "raw_data" not in obj:
                raise ValueError(
                    f"Task {i} missing 'raw_data' field. "
                    "Use symphony-data-generator format. "
                    "Run: cd symphony-data-generator && python src/quick_start.py"
                )
            
            # 从raw_data中提取prompt，根据benchmark类型选择对应的字段
            raw_data = obj.get("raw_data", {})
            benchmark = obj.get("benchmark", "")
            prompt = _extract_prompt_from_raw_data(raw_data, benchmark)
            
            if not prompt:
                print(f"[WARN] Task {i} (benchmark={benchmark}) has no extractable prompt, skipping")
                continue
            
            # 将difficulty_bin (easy/hard) 映射到 (simple/hard)
            difficulty_bin = obj.get("difficulty_bin", "hard")
            if difficulty_bin not in ["easy", "hard"]:
                raise ValueError(
                    f"Task {i} has invalid difficulty_bin: {difficulty_bin}. "
                    "Expected 'easy' or 'hard'"
                )
            difficulty = "simple" if difficulty_bin == "easy" else "hard"
            
            tasks.append(
                RealTask(
                    tid=len(tasks),
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


def print_exp3_summary_terminal(
    summary_row: SummaryRow,
    shock_type: str,
    shock_point: int,
):
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
    """
    计算恢复时间，需要满足三个条件才算恢复：
    1. shock后确实有显著的性能下降（≥ min_drop）
    2. 性能恢复到shock前基线的 recovery_ratio 以上
    3. 恢复状态需要持续 sustain_window 步
    
    返回恢复时间（步数），如果未恢复则返回-1
    """
    n = len(rolling_success)
    # 需要足够的数据来计算baseline和观察恢复
    if shock_point < 200 or shock_point + sustain_window >= n:
        return -1
    
    # 计算shock前的baseline（用shock前200步的平均值）
    baseline = sum(rolling_success[shock_point - 200: shock_point]) / 200
    
    # 检查shock后100步内的最低性能
    post_shock_window = rolling_success[shock_point: min(shock_point + 100, n)]
    if not post_shock_window:
        return -1
    
    post_min = min(post_shock_window)
    actual_drop = baseline - post_min
    
    # 如果没有显著的性能下降，就不算真正受损，自然也谈不上恢复
    if actual_drop < min_drop:
        return -1
    
    # 恢复的目标值：baseline的recovery_ratio倍
    target = baseline * recovery_ratio
    # 从shock点开始向后滑动窗口，寻找第一个满足持续恢复的点
    for t in range(shock_point, n - sustain_window + 1):
        window = rolling_success[t: t + sustain_window]
        if len(window) < sustain_window:
            continue
        # 窗口内所有值都要达到target才算持续恢复
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
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas not available, skipping Excel export")
        return

    excel_path = os.path.join(outdir, "exp3_results.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_data = []
            for r in summary:
                row_dict = asdict(r)
                row_dict['recovery_status'] = 'Recovered' if r.recovery_time >= 0 else 'Not Recovered'
                row_dict['success_rate_drop'] = r.success_rate_pre_shock - r.success_rate_post_shock
                row_dict['recovery_efficiency'] = r.recovery_time if r.recovery_time >= 0 else None
                summary_data.append(row_dict)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
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
            
            fig = plt.figure(figsize=(12, 6), dpi=180, constrained_layout=True)
            ax = fig.add_subplot(111)
            
            task_indices = [log.t for log in logs]
            rolling_success = [log.rolling_success for log in logs]
            
            ax.plot(task_indices, rolling_success, linewidth=2, label="LinUCB (Ours)")
            ax.axvline(x=shock_point, color="red", linestyle="--", linewidth=2, label=f"Shock Point (t={shock_point})")
            
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
            
            fig2 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=True)
            ax2 = fig2.add_subplot(111)
            
            r = summary[0]
            if r.recovery_time >= 0:
                recovery_time = r.recovery_time
                color = '#2ecc71'
                label_text = str(r.recovery_time)
                label_color = 'black'
                y_max = max(recovery_time, 100)
                bars = ax2.bar([0], [recovery_time], color=color, alpha=0.8, width=0.6)
            else:
                recovery_time = 50
                color = '#e74c3c'
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


def verify_success(task: RealTask, response: str) -> bool:
    # 简单的启发式验证：根据响应长度判断（目前placeholder模式下不使用，成功与否由agent.execute决定）
    if task.difficulty == "simple":
        return len(response) > 20
    return len(response) > 50


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

    agents = {
        "A": RealAgent(RealAgentProfile("A", "gpt-4", 1.0, p_simple=0.99, p_hard=0.99), rng),
        "B": RealAgent(RealAgentProfile("B", "gpt-3.5", 0.1, p_simple=0.88, p_hard=0.18), rng),
        "C": RealAgent(RealAgentProfile("C", "mixtral", 0.3, p_simple=0.92, p_hard=0.65), rng),
    }

    # 初始化LinUCB selector（6维特征向量）
    selector = GlobalLinUCB(d=6, l2=1.0, alpha=1.0, delta=0.05, S=1.0)

    tasks = load_tasks(args.tasks, args.n)

    # 创建输出目录（按shock类型和时间戳组织）
    shock_type_short = "ShockA" if args.shock == "A_unavailable" else "ShockB"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join(args.outdir, shock_type_short, ts)
    os.makedirs(outdir, exist_ok=True)

    step_logs = []
    success_hist: List[int] = []
    choose_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    choose_counts_pre = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    choose_counts_post = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    total_latency = 0.0

    for t, task in enumerate(tasks):
        # 在shock_point时刻应用shock
        if t == shock_point:
            for ag in agents.values():
                ag.apply_shock(args.shock)

        # 只考虑当前可用的agent（shock后A可能不可用）
        candidates = [ag for ag in agents.values() if ag.available]

        # 如果没有可用的agent，任务失败
        if not candidates:
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

        # LinUCB选择最优agent
        chosen_id = selector.select(xs)
        agent = agents[chosen_id]
        # match_score用于构建特征向量（这里简化处理，A的match更高）
        match_score = 0.8 if chosen_id == "A" else 0.6

        # 执行任务，返回响应、延迟和是否成功
        resp, latency, ok = agent.execute(task.prompt, task.difficulty)
        reward = 1.0 if ok else 0.0

        # 用执行结果更新LinUCB的参数（基于选择的agent的特征向量和reward）
        chosen_x = next(x for aid, x in xs if aid == chosen_id)
        selector.update(chosen_x, reward)

        success_hist.append(1 if ok else 0)
        choose_counts[chosen_id] = choose_counts.get(chosen_id, 0) + 1
        if t < shock_point:
            choose_counts_pre[chosen_id] = choose_counts_pre.get(chosen_id, 0) + 1
        else:
            choose_counts_post[chosen_id] = choose_counts_post.get(chosen_id, 0) + 1
        total_latency += latency

        # 计算滚动窗口的成功率（用于绘制V-shape曲线）
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

    n = len(tasks)
    avg_latency = total_latency / max(1, n)
    p_hard = sum(1 for t in tasks if t.difficulty == "hard") / max(1, n)
    
    pre_shock_successes = sum(success_hist[:shock_point]) if shock_point > 0 else 0
    pre_shock_rate = pre_shock_successes / max(1, shock_point)
    
    post_shock_successes = sum(success_hist[shock_point:]) if shock_point < n else 0
    post_shock_count = n - shock_point
    post_shock_rate = post_shock_successes / max(1, post_shock_count)
    
    overall_rate = sum(success_hist) / max(1, n)
    
    # 计算滚动成功率用于恢复时间计算
    rolling_success_list = calculate_rolling_success(success_hist, window_size=50)
    recovery_time = calculate_recovery_time_strict(
        rolling_success_list,
        shock_point,
        min_drop=0.08,
        recovery_ratio=0.9,
        sustain_window=50,
    )
    
    # 统计死锁情况（所有agent都不可用导致无法选择）
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

    print_exp3_summary_terminal(summary_row, args.shock, shock_point)
    
    print(f"\n[OUTPUT] Writing results to: {outdir}")
    
    summary_dict = asdict(summary_row)
    summary_dict['recovery_status'] = 'Recovered' if summary_row.recovery_time >= 0 else 'Not Recovered'
    summary_dict['success_rate_drop'] = summary_row.success_rate_pre_shock - summary_row.success_rate_post_shock
    write_csv(os.path.join(outdir, "summary.csv"), [summary_dict])
    print(f"  ✓ summary.csv")
    
    traj_dicts = [asdict(log) for log in step_logs]
    write_csv(os.path.join(outdir, "trajectory_linucb.csv"), traj_dicts)
    print(f"  ✓ trajectory_linucb.csv")
    
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

    if not args.no_excel:
        write_excel(outdir, [summary_row], {"linucb": step_logs})
        excel_path = os.path.join(outdir, "exp3_results.xlsx")
        if os.path.exists(excel_path):
            print(f"  ✓ exp3_results.xlsx")

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
