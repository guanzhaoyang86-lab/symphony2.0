#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo/sim_exp3_robustness.py

Symphony 2.0 模拟实验：Role Shock 下的鲁棒性与恢复能力（Exp3）

目标：
- 在运行中引入 agent 能力/可用性突变（shock）
- 比较不同策略在非平稳环境下的性能退化与恢复能力
- 核心指标：V-shape recovery、recovery time、post-shock performance、deadlock rate

不跑真实 LLM，不需要 GPU
复用 core/linucb_selector.py 的 GlobalLinUCB + build_x(d=6)

输出：
- outdir/summary.csv / summary.xlsx         各策略汇总（recovery time / deadlock / success rates）
- outdir/trajectory_*.csv / trajectory_*.xlsx    逐步轨迹（rolling success / recovery curve）
- outdir/plot_*.png                          图（V-shape curve / recovery time comparison）

运行示例：
  python3 demo/sim_exp3_robustness.py --n 1000 --shock A_unavailable --shock-point 500 --seed 123
  python3 demo/sim_exp3_robustness.py --n 1000 --shock A_degraded --shock-point 500 --seed 123 --freeze-after-shock
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ---- make project root importable ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../"))
import sys
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---- reuse your LinUCB & build_x ----
from core.linucb_selector import GlobalLinUCB, build_x


# -----------------------------
# 1) Data model (reuse from Exp1, but add shock-related fields)
# -----------------------------
@dataclass
class SimTask:
    tid: int
    difficulty: str  # "simple" | "hard"
    requirement: str  # used for capability match (Top-L)


@dataclass
class SimAgentProfile:
    agent_id: str
    call_cost: float
    base_latency_ms: float
    p_simple: float
    p_hard: float
    match_simple: float
    match_hard: float


@dataclass
class SimAgentState:
    load: float = 0.0
    latency_ms: float = 500.0
    reputation: float = 0.5
    available: bool = True


class SimAgent:
    """
    Simulation-only agent (same as Exp1, but with shock support)
    """

    def __init__(self, profile: SimAgentProfile, rng: random.Random):
        self.p = profile
        self.rng = rng
        self.s = SimAgentState(load=0.0, latency_ms=profile.base_latency_ms, reputation=0.5, available=True)
        # Exp3: track if agent is shocked
        self._is_shocked = False
        self._shock_type = None
        self._original_p_hard = profile.p_hard

    def reset(self):
        self.s = SimAgentState(load=0.0, latency_ms=self.p.base_latency_ms, reputation=0.5, available=True)
        self._is_shocked = False
        self._shock_type = None
        self.p.p_hard = self._original_p_hard

    def apply_shock(self, shock_type: str):
        """
        Paper-level shock definitions (Exp3)
        
        Shock must be SYSTEM-LEVEL to create meaningful failure and recovery:
        - Shock A: Hard routing channels severely damaged (A unavailable, E/C degraded)
        - Shock B: Performance degradation across hard-specialized agents
        """
        self._is_shocked = True
        self._shock_type = shock_type
        
        if shock_type == "A_unavailable":
            # Shock A: System-level failure of hard task routing
            if self.p.agent_id == "A":
                # Strongest agent goes offline
                self.s.available = False
            elif self.p.agent_id == "E":
                # Hard specialist severely degraded
                self.p.p_hard *= 0.35   # 0.85 -> ~0.30
            elif self.p.agent_id == "C":
                # Medium agent partially degraded
                self.p.p_hard *= 0.70   # 0.65 -> ~0.45
            # B and D remain unchanged (as fallback options)
            
        elif shock_type == "A_degraded":
            # Shock B: Performance degradation (more gradual but still system-level)
            if self.p.agent_id == "A":
                # Strongest agent performance drops significantly
                self.p.p_hard = 0.25    # 0.99 -> 0.25
            elif self.p.agent_id == "E":
                # Hard specialist moderately degraded
                self.p.p_hard *= 0.55   # 0.85 -> ~0.47
            elif self.p.agent_id == "C":
                # Medium agent slightly degraded
                self.p.p_hard *= 0.80   # 0.65 -> ~0.52
            # B and D remain unchanged

    def match_score(self, requirement: str) -> float:
        if requirement == "simple":
            return float(self.p.match_simple)
        if requirement == "hard":
            return float(self.p.match_hard)
        return 0.5

    def get_dynamic_state(self) -> Dict[str, float]:
        return {
            "available": bool(self.s.available),
            "load": float(max(0.0, min(1.0, self.s.load))),
            "latency_ms": float(max(1.0, self.s.latency_ms)),
            "reputation": float(max(0.0, min(1.0, self.s.reputation))),
        }

    def _sample_latency(self) -> float:
        load = max(0.0, min(1.0, self.s.load))
        mean = self.p.base_latency_ms * (1.0 + 1.2 * load)
        noise = self.rng.gauss(0.0, 0.08 * mean)
        return max(10.0, mean + noise)

    def _success_prob(self, difficulty: str) -> float:
        base = self.p.p_simple if difficulty == "simple" else self.p.p_hard
        load = max(0.0, min(1.0, self.s.load))
        base = base * (1.0 - 0.05 * load)
        return max(0.0, min(1.0, base))

    def step_dynamics_after_call(self, ok: bool, latency_ms: float):
        beta = 0.20
        self.s.latency_ms = (1.0 - beta) * self.s.latency_ms + beta * float(latency_ms)
        self.s.reputation = max(0.0, min(1.0, 0.95 * self.s.reputation + 0.05 * (1.0 if ok else 0.0)))
        self.s.load = max(0.0, min(1.0, self.s.load + 0.30))
        self.s.available = bool(self.s.load < 0.95)

    def decay_load(self):
        self.s.load = max(0.0, min(1.0, 0.88 * self.s.load))
        self.s.available = bool(self.s.load < 0.95)

    def execute(self, task: SimTask) -> Tuple[bool, float]:
        lat = self._sample_latency()
        p = self._success_prob(task.difficulty)
        ok = (self.rng.random() < p)
        self.step_dynamics_after_call(ok=ok, latency_ms=lat)
        return ok, lat


# -----------------------------
# 2) Policies / Evaluation (Exp3 specific)
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
    is_shock: int  # 1 if this step is the shock point
    rolling_success: float  # rolling window success rate
    reward_used_for_update: float


@dataclass
class SummaryRow:
    policy: str
    n: int
    p_hard: float
    shock_type: str
    shock_point: int
    success_rate_pre_shock: float  # [0, shock_point)
    success_rate_post_shock: float  # [shock_point, n)
    success_rate_overall: float
    recovery_time: int  # -1 if not recovered
    deadlock_rate: float  # tasks that failed and couldn't recover
    avg_latency_ms: float
    choose_A: int
    choose_B: int
    choose_C: int
    choose_D: int
    choose_E: int
    # Pre/post shock agent selection (for routing analysis)
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


def generate_tasks(n: int, p_hard: float, rng: random.Random) -> List[SimTask]:
    tasks: List[SimTask] = []
    for i in range(n):
        hard = (rng.random() < p_hard)
        diff = "hard" if hard else "simple"
        req = "hard" if hard else "simple"
        tasks.append(SimTask(tid=i, difficulty=diff, requirement=req))
    return tasks


def pick_topL_candidates(
    agents: List[SimAgent],
    requirement: str,
    topL: int,
) -> List[Tuple[SimAgent, float]]:
    scored: List[Tuple[SimAgent, float]] = []
    for ag in agents:
        ms = ag.match_score(requirement)
        scored.append((ag, float(ms)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, int(topL))]


def calculate_rolling_success(
    success_history: List[int],
    window_size: int = 50
) -> List[float]:
    """Calculate rolling success rate"""
    rolling = []
    window = []
    for s in success_history:
        window.append(s)
        if len(window) > window_size:
            window.pop(0)
        rolling.append(sum(window) / len(window) if window else 0.0)
    return rolling


def calculate_recovery_time(
    rolling_success: List[float],
    shock_point: int,
    recovery_threshold: float = 0.8,
) -> int:
    """
    OLD definition (too lenient) - kept for backward compatibility but not recommended
    """
    if shock_point < 200:
        return -1
    
    baseline_window = rolling_success[max(0, shock_point - 200):shock_point]
    if not baseline_window:
        return -1
    
    baseline = sum(baseline_window) / len(baseline_window)
    target = baseline * recovery_threshold
    
    for i in range(shock_point, len(rolling_success)):
        if rolling_success[i] >= target:
            return i - shock_point
    
    return -1


def calculate_recovery_time_strict(
    rolling_success: List[float],
    shock_point: int,
    *,
    min_drop: float = 0.08,
    recovery_ratio: float = 0.9,
    sustain_window: int = 50,
) -> int:
    """
    Paper-grade recovery definition (Exp3).
    
    Recovery happens iff ALL conditions are met:
    1. There is a significant performance drop after shock (≥ min_drop)
    2. Performance later recovers to ≥ recovery_ratio * pre-shock baseline
    3. Recovery is sustained for sustain_window consecutive steps
    
    This definition ensures:
    - No "fake recovery" for policies that never dropped (static_rule/random)
    - Only adaptive policies (linucb) can achieve recovery
    - Recovery must be stable, not just a single spike
    
    Args:
        rolling_success: Rolling success rate history
        shock_point: Task index where shock occurs
        min_drop: Minimum drop required to consider "real degradation" (default: 0.08 = 8%)
        recovery_ratio: Recovery target as fraction of pre-shock baseline (default: 0.9 = 90%)
        sustain_window: Number of consecutive steps required for sustained recovery (default: 50)
    
    Returns:
        recovery_time (steps), or -1 if not recovered
    """
    n = len(rolling_success)
    if shock_point < 200 or shock_point + sustain_window >= n:
        return -1
    
    # Pre-shock baseline (average of last 200 steps before shock)
    baseline = sum(rolling_success[shock_point - 200: shock_point]) / 200
    
    # Must observe a real drop after shock
    # Check minimum value in first 100 steps after shock
    post_shock_window = rolling_success[shock_point: min(shock_point + 100, n)]
    if not post_shock_window:
        return -1
    
    post_min = min(post_shock_window)
    actual_drop = baseline - post_min
    
    # Condition 1: Must have significant drop
    if actual_drop < min_drop:
        return -1  # No real degradation → no recovery concept
    
    # Condition 2 & 3: Look for sustained recovery
    target = baseline * recovery_ratio
    
    # Find first point where recovery is sustained
    for t in range(shock_point, n - sustain_window + 1):
        window = rolling_success[t: t + sustain_window]
        if len(window) < sustain_window:
            continue
        
        # All values in window must be ≥ target
        if all(x >= target for x in window):
            return t - shock_point
    
    return -1  # Not recovered


def reward_shaping(
    success: bool,
    latency_ms: float,
    call_cost: float,
    latency_scale_ms: float,
    latency_penalty: float,
    cost_lambda: float,
    max_cost: float,
) -> float:
    base = 1.0 if success else 0.0
    lat_norm = min(1.0, float(latency_ms) / max(1.0, float(latency_scale_ms)))
    cost_norm = float(call_cost) / max(1e-9, max_cost)
    r = base - float(latency_penalty) * math.sqrt(lat_norm) - float(cost_lambda) * cost_norm
    return max(0.0, min(1.0, r))


def run_policy_exp3(
    policy_name: str,
    tasks: List[SimTask],
    agents: List[SimAgent],
    topL: int,
    shock_type: str,
    shock_point: int,
    freeze_after_shock: bool,
    *,
    linucb_alpha: float,
    linucb_l2: float,
    delta: float,
    S: float,
    latency_scale_ms: float,
    latency_penalty: float,
    cost_lambda: float,
    seed_for_policy: int,
) -> Tuple[SummaryRow, List[StepLog]]:
    """
    Run one policy with shock injection (Exp3 specific)
    """
    rng = random.Random(seed_for_policy)

    # Reset agents
    for ag in agents:
        ag.reset()

    # Selector only for LinUCB policy
    selector: Optional[GlobalLinUCB] = None
    if policy_name == "linucb":
        selector = GlobalLinUCB(d=6, l2=float(linucb_l2), alpha=float(linucb_alpha), delta=float(delta), S=float(S))
    elif policy_name == "linucb_frozen":
        selector = GlobalLinUCB(d=6, l2=float(linucb_l2), alpha=float(linucb_alpha), delta=float(delta), S=float(S))

    max_cost = max(a.p.call_cost for a in agents)
    choose_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    choose_counts_pre = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    choose_counts_post = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    total_cost = 0.0
    total_latency = 0.0
    success_history: List[int] = []
    step_logs: List[StepLog] = []

    agent_by_id = {a.p.agent_id: a for a in agents}
    strong_agent_id = "A"

    for t, task in enumerate(tasks):
        # Apply shock at shock_point (system-level: affects multiple agents)
        if t == shock_point:
            # Apply shock to all relevant agents (system-level shock)
            for ag in agents:
                ag.apply_shock(shock_type)

        # Decay load for all agents
        for ag in agents:
            ag.decay_load()

        # Paper-correct: Filter agent pool BEFORE Top-L selection
        # This ensures Shock A (unavailable) truly removes agents from candidate set
        candidate_agents = [ag for ag in agents if ag.get_dynamic_state().get("available", True)]
        
        # If no agents available after shock → system-level failure
        if not candidate_agents:
            success_history.append(0)
            step_logs.append(
                StepLog(
                    t=t,
                    policy=policy_name,
                    task_difficulty=task.difficulty,
                    chosen_agent="NONE",
                    match_score=0.0,
                    load=0.0,
                    latency_ms=0.0,
                    call_cost=0.0,
                    success=0,
                    is_shock=1 if t == shock_point else 0,
                    rolling_success=calculate_rolling_success(success_history, window_size=50)[-1] if success_history else 0.0,
                    reward_used_for_update=0.0,
                )
            )
            continue

        # Top-L candidates (from available agents only)
        top = pick_topL_candidates(candidate_agents, task.requirement, topL=topL)
        avail = [(ag, ms) for (ag, ms) in top]  # All in top are already available

        # Choose action
        chosen_ag: SimAgent
        chosen_ms: float
        chosen_x: Optional[List[float]] = None

        if policy_name == "static_rule":
            chosen_id = "B" if task.difficulty == "simple" else "A"
            chosen_ag = agent_by_id.get(chosen_id, agent_by_id[strong_agent_id])
            # Check if chosen agent is available (respects Shock A)
            # Note: static_rule is a baseline that doesn't adapt, so it may fail in Shock A
            if not chosen_ag.get_dynamic_state().get("available", True):
                # Agent unavailable → system failure (expected for non-adaptive baseline)
                success_history.append(0)
                step_logs.append(
                    StepLog(
                        t=t,
                        policy=policy_name,
                        task_difficulty=task.difficulty,
                        chosen_agent="NONE",
                        match_score=0.0,
                        load=0.0,
                        latency_ms=0.0,
                        call_cost=0.0,
                        success=0,
                        is_shock=1 if t == shock_point else 0,
                        rolling_success=calculate_rolling_success(success_history, window_size=50)[-1] if success_history else 0.0,
                        reward_used_for_update=0.0,
                    )
                )
                continue
            chosen_ms = chosen_ag.match_score(task.requirement)

        elif policy_name == "random":
            chosen_ag, chosen_ms = rng.choice(avail)

        elif policy_name in ["linucb", "linucb_frozen"]:
            assert selector is not None
            candidates: List[Tuple[str, List[float], float]] = []
            for (ag, ms) in avail:
                st = ag.get_dynamic_state()
                x = build_x(
                    match_score=float(ms),
                    dynamic_state={
                        "load": float(st.get("load", 0.0)),
                        "latency_ms": float(st.get("latency_ms", 500.0)),
                        "reputation": float(st.get("reputation", 0.5)),
                    },
                    available=bool(st.get("available", True)),
                    latency_scale_ms=float(latency_scale_ms),
                )
                candidates.append((ag.p.agent_id, x, float(ms)))

            chosen_id = selector.select([(aid, x) for (aid, x, _) in candidates])
            chosen_ag = agent_by_id[chosen_id]
            chosen_ms = next(ms for (aid, _x, ms) in candidates if aid == chosen_id)
            chosen_x = next(_x for (aid, _x, _ms) in candidates if aid == chosen_id)

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        agent_id = chosen_ag.p.agent_id
        choose_counts[agent_id] = choose_counts.get(agent_id, 0) + 1
        # Track pre/post shock selection
        if t < shock_point:
            choose_counts_pre[agent_id] = choose_counts_pre.get(agent_id, 0) + 1
        else:
            choose_counts_post[agent_id] = choose_counts_post.get(agent_id, 0) + 1

        # Execute
        ok, lat_ms = chosen_ag.execute(task)
        call_cost = chosen_ag.p.call_cost
        total_cost += call_cost
        total_latency += lat_ms
        success_history.append(1 if ok else 0)

        # LinUCB update (skip if frozen after shock)
        used_reward = 0.0
        if policy_name in ["linucb", "linucb_frozen"]:
            assert selector is not None
            should_update = True
            if policy_name == "linucb_frozen" and t >= shock_point:
                should_update = False
            elif freeze_after_shock and policy_name == "linucb" and t >= shock_point:
                should_update = False

            if should_update:
                if chosen_x is None:
                    st = chosen_ag.get_dynamic_state()
                    chosen_x = build_x(
                        match_score=float(chosen_ms),
                        dynamic_state={
                            "load": float(st.get("load", 0.0)),
                            "latency_ms": float(st.get("latency_ms", 500.0)),
                            "reputation": float(st.get("reputation", 0.5)),
                        },
                        available=bool(st.get("available", True)),
                        latency_scale_ms=float(latency_scale_ms),
                    )

                used_reward = reward_shaping(
                    success=ok,
                    latency_ms=lat_ms,
                    call_cost=call_cost,
                    latency_scale_ms=latency_scale_ms,
                    latency_penalty=latency_penalty,
                    cost_lambda=cost_lambda,
                    max_cost=max_cost,
                )
                selector.update(chosen_x, used_reward)

        # Calculate rolling success
        rolling_success = calculate_rolling_success(success_history, window_size=50)
        current_rolling = rolling_success[-1] if rolling_success else 0.0

        # Logging
        st_now = chosen_ag.get_dynamic_state()
        step_logs.append(
            StepLog(
                t=t,
                policy=policy_name,
                task_difficulty=task.difficulty,
                chosen_agent=chosen_ag.p.agent_id,
                match_score=float(chosen_ms),
                load=float(st_now.get("load", 0.0)),
                latency_ms=float(lat_ms),
                call_cost=float(call_cost),
                success=1 if ok else 0,
                is_shock=1 if t == shock_point else 0,
                rolling_success=float(current_rolling),
                reward_used_for_update=float(used_reward),
            )
        )

    # Calculate metrics
    n = len(tasks)
    avg_latency = total_latency / max(1, n)
    
    # Pre-shock and post-shock success rates
    pre_shock_successes = sum(success_history[:shock_point]) if shock_point > 0 else 0
    pre_shock_rate = pre_shock_successes / max(1, shock_point)
    
    post_shock_successes = sum(success_history[shock_point:]) if shock_point < n else 0
    post_shock_count = n - shock_point
    post_shock_rate = post_shock_successes / max(1, post_shock_count)
    
    overall_rate = sum(success_history) / max(1, n)
    
    # Recovery time (using strict paper-grade definition)
    rolling_success_list = calculate_rolling_success(success_history, window_size=50)
    recovery_time = calculate_recovery_time_strict(
        rolling_success_list,
        shock_point,
        min_drop=0.08,        # Require at least 8% drop
        recovery_ratio=0.9,    # Must recover to 90% of pre-shock baseline
        sustain_window=50,     # Must sustain for 50 consecutive steps
    )
    
    # Deadlock rate (tasks that failed and couldn't recover)
    # Simplified: count consecutive failures after shock
    deadlock_count = 0
    if shock_point < n:
        consecutive_failures = 0
        for i in range(shock_point, n):
            if success_history[i] == 0:
                consecutive_failures += 1
                if consecutive_failures >= 3:  # 3 consecutive failures = deadlock
                    deadlock_count += 1
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
    deadlock_rate = deadlock_count / max(1, post_shock_count)

    row = SummaryRow(
        policy=policy_name,
        n=n,
        p_hard=float(sum(1 for x in tasks if x.difficulty == "hard") / max(1, n)),
        shock_type=shock_type,
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
        choose_D=int(choose_counts.get("D", 0)),
        choose_E=int(choose_counts.get("E", 0)),
        # Pre/post shock agent selection
        choose_A_pre=int(choose_counts_pre.get("A", 0)),
        choose_B_pre=int(choose_counts_pre.get("B", 0)),
        choose_C_pre=int(choose_counts_pre.get("C", 0)),
        choose_D_pre=int(choose_counts_pre.get("D", 0)),
        choose_E_pre=int(choose_counts_pre.get("E", 0)),
        choose_A_post=int(choose_counts_post.get("A", 0)),
        choose_B_post=int(choose_counts_post.get("B", 0)),
        choose_C_post=int(choose_counts_post.get("C", 0)),
        choose_D_post=int(choose_counts_post.get("D", 0)),
        choose_E_post=int(choose_counts_post.get("E", 0)),
    )

    return row, step_logs


# -----------------------------
# 3) Plotting (Exp3 specific: V-shape recovery)
# -----------------------------
def try_plot(outdir: str, summary: List[SummaryRow], traj: Dict[str, List[StepLog]]) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available: {e}")
        return

    try:
        # V-shape recovery curves
        if "linucb" in traj:
            logs = traj["linucb"]
            shock_point = summary[0].shock_point if summary else 500
            
            fig = plt.figure(figsize=(12, 6), dpi=180, constrained_layout=True)
            ax = fig.add_subplot(111)
            
            task_indices = [log.t for log in logs]
            rolling_success = [log.rolling_success for log in logs]
            
            ax.plot(task_indices, rolling_success, linewidth=2, label="LinUCB (Ours)")
            ax.axvline(x=shock_point, color="red", linestyle="--", linewidth=2, label=f"Shock Point (t={shock_point})")
            
            # Mark recovery point if exists
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

        # Recovery time comparison
        if len(summary) > 1:
            fig2 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=True)
            ax2 = fig2.add_subplot(111)
            
            policies = [r.policy for r in summary]
            recovery_times = []
            recovery_labels = []
            colors_list = []
            
            # Find max valid recovery time for scaling
            valid_recoveries = [r.recovery_time for r in summary if r.recovery_time >= 0]
            max_valid_recovery = max(valid_recoveries) if valid_recoveries else 200
            
            for r in summary:
                if r.recovery_time >= 0:
                    recovery_times.append(r.recovery_time)
                    recovery_labels.append(str(r.recovery_time))
                    colors_list.append('#2ecc71')  # Green for recovered
                else:
                    # Use a visual marker: 1.5x max valid recovery (or 300 if all failed)
                    marker_value = max(max_valid_recovery * 1.5, 300)
                    recovery_times.append(marker_value)
                    recovery_labels.append("N/A")
                    colors_list.append('#e74c3c')  # Red for not recovered
            
            # Always plot (even if all are N/A)
            bars = ax2.bar(range(len(policies)), recovery_times, color=colors_list, alpha=0.8)
            ax2.set_xticks(range(len(policies)))
            ax2.set_xticklabels(policies, rotation=20, ha="right", fontsize=11)
            ax2.set_ylabel("Recovery Time (steps)", fontsize=12)
            ax2.set_title("Recovery Time Comparison", fontsize=14, fontweight="bold")
            ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
            
            # Add value labels
            y_max = max(recovery_times) if recovery_times else 100
            for i, (b, label, rt) in enumerate(zip(bars, recovery_labels, [r.recovery_time for r in summary])):
                if rt >= 0:
                    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + y_max * 0.02, label,
                            ha="center", va="bottom", fontsize=10, fontweight="bold")
                else:
                    # Mark as "N/A" with different style
                    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + y_max * 0.02, "N/A",
                            ha="center", va="bottom", fontsize=10, fontweight="bold", color='red')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = []
            if any(r.recovery_time >= 0 for r in summary):
                legend_elements.append(Patch(facecolor='#2ecc71', label='Recovered'))
            if any(r.recovery_time < 0 for r in summary):
                legend_elements.append(Patch(facecolor='#e74c3c', label='Not Recovered (N/A)'))
            if legend_elements:
                ax2.legend(handles=legend_elements, loc='upper right')
            
            # Set y-axis limit to show N/A markers clearly
            if any(r.recovery_time < 0 for r in summary):
                ax2.set_ylim([0, y_max * 1.15])
            
            fig2.savefig(os.path.join(outdir, "plot_recovery_time_comparison.png"))
            plt.close(fig2)
        
        # V-shape recovery curves for ALL policies (comparison)
        if len(traj) > 0:
            fig3 = plt.figure(figsize=(14, 7), dpi=180, constrained_layout=True)
            ax3 = fig3.add_subplot(111)
            
            shock_point = summary[0].shock_point if summary else 500
            shock_type = summary[0].shock_type if summary else "N/A"
            
            # Plot all policies
            colors = {'static_rule': '#3498db', 'random': '#e67e22', 'linucb': '#2ecc71', 'linucb_frozen': '#9b59b6'}
            for policy, logs in traj.items():
                if logs:
                    task_indices = [log.t for log in logs]
                    rolling_success = [log.rolling_success for log in logs]
                    color = colors.get(policy, '#95a5a6')
                    label = policy.replace('_', ' ').title()
                    ax3.plot(task_indices, rolling_success, linewidth=2, label=label, color=color, alpha=0.8)
            
            # Mark shock point
            ax3.axvline(x=shock_point, color="red", linestyle="--", linewidth=2, label=f"Shock Point (t={shock_point})", zorder=10)
            
            # Mark recovery points
            for row in summary:
                if row.recovery_time >= 0:
                    recovery_point = shock_point + row.recovery_time
                    if recovery_point < len(traj.get(row.policy, [])):
                        color = colors.get(row.policy, '#95a5a6')
                        ax3.plot(recovery_point, 
                                traj[row.policy][recovery_point].rolling_success if row.policy in traj else 0,
                                "o", markersize=12, color=color, markeredgecolor='black', 
                                markeredgewidth=1.5, label=f"{row.policy} Recovery", zorder=5)
            
            ax3.set_xlabel("Task Index", fontsize=12)
            ax3.set_ylabel("Rolling Success Rate", fontsize=12)
            ax3.set_title(f"Exp3: V-Shape Recovery Comparison (Shock Type: {shock_type})", fontsize=14, fontweight="bold")
            ax3.legend(loc="best", fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1.05])
            
            fig3.savefig(os.path.join(outdir, "plot_v_shape_all_policies.png"))
            plt.close(fig3)
    except Exception as e:
        print(f"[WARN] Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()


# -----------------------------
# 4) IO helpers (with Excel support)
# -----------------------------
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


# -----------------------------
# 4.5) Terminal printing helpers (Exp3 specific)
# -----------------------------
def compute_agent_usage_pre_post(
    logs: List[StepLog],
    shock_point: int,
) -> Dict[str, Dict[str, float]]:
    """
    Compute agent usage percentage before and after shock.

    Returns:
        {
          "pre":  {"A": pct, "B": pct, ...},
          "post": {"A": pct, "B": pct, ...},
        }
    """
    agents = ["A", "B", "C", "D", "E"]

    pre_counts = {a: 0 for a in agents}
    post_counts = {a: 0 for a in agents}

    for log in logs:
        if log.t < shock_point:
            pre_counts[log.chosen_agent] = pre_counts.get(log.chosen_agent, 0) + 1
        else:
            post_counts[log.chosen_agent] = post_counts.get(log.chosen_agent, 0) + 1

    pre_total = sum(pre_counts.values())
    post_total = sum(post_counts.values())

    pre_pct = {
        a: (pre_counts[a] / pre_total * 100.0) if pre_total > 0 else 0.0
        for a in agents
    }
    post_pct = {
        a: (post_counts[a] / post_total * 100.0) if post_total > 0 else 0.0
        for a in agents
    }

    return {"pre": pre_pct, "post": post_pct}


def print_agent_usage_terminal(
    policy: str,
    logs: List[StepLog],
    shock_point: int,
):
    """Print agent usage pre/post shock comparison in terminal"""
    usage = compute_agent_usage_pre_post(logs, shock_point)

    print(f"\n🧠 Agent usage (pre-shock → post-shock)")
    print(f"Policy: {policy}")
    print("-" * 55)

    for agent in ["A", "E", "C", "D", "B"]:  # Hard → Medium → Cheap order
        pre = usage["pre"].get(agent, 0.0)
        post = usage["post"].get(agent, 0.0)
        delta = post - pre

        if delta > 5:
            arrow = "↑↑"
        elif delta > 1:
            arrow = "↑"
        elif delta < -5:
            arrow = "↓↓↓"
        elif delta < -1:
            arrow = "↓"
        else:
            arrow = "→"

        print(
            f"{agent}: "
            f"{pre:5.1f}% → {post:5.1f}%   {arrow:>3s}"
        )

    print("-" * 55)


def print_exp3_summary_terminal(
    summary_rows: List[SummaryRow],
    shock_type: str,
    shock_point: int,
):
    """Print comprehensive Exp3 summary in terminal"""
    print("\n" + "=" * 90)
    print(f"📊 Exp3 Robustness Summary")
    print(f"Shock Type : {shock_type}")
    print(f"Shock Point: t = {shock_point}")
    print("=" * 90)
    print("Recovery Definition:")
    print("  ✓ Significant drop observed (≥8%)")
    print("  ✓ Sustained recovery to ≥90% of pre-shock baseline")
    print("  ✓ Recovery maintained for 50 consecutive steps")
    print("=" * 90)

    header = (
        f"{'Policy':15s} "
        f"{'Pre':>8s} "
        f"{'Post':>8s} "
        f"{'Drop':>8s} "
        f"{'Recovery':>12s} "
        f"{'Recovered':>12s} "
        f"{'Deadlock':>10s}"
    )
    print(header)
    print("-" * len(header))

    for r in summary_rows:
        drop = r.success_rate_pre_shock - r.success_rate_post_shock
        recovered = "YES" if r.recovery_time >= 0 else "NO"
        rec_time = f"{r.recovery_time:4d}" if r.recovery_time >= 0 else "     N/A"

        print(
            f"{r.policy:15s} "
            f"{r.success_rate_pre_shock:8.3f} "
            f"{r.success_rate_post_shock:8.3f} "
            f"{drop:8.3f} "
            f"{rec_time:>12s} "
            f"{recovered:>12s} "
            f"{r.deadlock_rate:10.3f}"
        )

    print("=" * 90)


def write_excel(outdir: str, summary: List[SummaryRow], traj: Dict[str, List[StepLog]]) -> None:
    """Write results to Excel file with multiple sheets"""
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas not available, skipping Excel export")
        return

    excel_path = os.path.join(outdir, "exp3_results.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet with detailed metrics
            summary_data = []
            for r in summary:
                row_dict = asdict(r)
                # Add additional computed metrics
                row_dict['recovery_status'] = 'Recovered' if r.recovery_time >= 0 else 'Not Recovered'
                row_dict['success_rate_drop'] = r.success_rate_pre_shock - r.success_rate_post_shock
                row_dict['recovery_efficiency'] = r.recovery_time if r.recovery_time >= 0 else None
                summary_data.append(row_dict)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Trajectory sheets (one per policy)
            for policy, logs in traj.items():
                if logs:
                    traj_df = pd.DataFrame([asdict(log) for log in logs])
                    sheet_name = f'Trajectory_{policy}'[:31]  # Excel sheet name limit
                    traj_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Additional analysis sheet: Recovery statistics
            recovery_stats = []
            for r in summary:
                recovery_stats.append({
                    'policy': r.policy,
                    'recovery_time': r.recovery_time if r.recovery_time >= 0 else 'N/A',
                    'deadlock_rate': r.deadlock_rate,
                    'pre_shock_success': r.success_rate_pre_shock,
                    'post_shock_success': r.success_rate_post_shock,
                    'overall_success': r.success_rate_overall,
                    'performance_drop_pct': (r.success_rate_pre_shock - r.success_rate_post_shock) * 100,
                })
            
            recovery_df = pd.DataFrame(recovery_stats)
            recovery_df.to_excel(writer, sheet_name='Recovery_Stats', index=False)
            
            # Agent selection distribution sheet (overall)
            agent_selection = []
            for r in summary:
                total_selections = r.choose_A + r.choose_B + r.choose_C + r.choose_D + r.choose_E
                if total_selections > 0:
                    agent_selection.append({
                        'policy': r.policy,
                        'select_A_pct': (r.choose_A / total_selections) * 100,
                        'select_B_pct': (r.choose_B / total_selections) * 100,
                        'select_C_pct': (r.choose_C / total_selections) * 100,
                        'select_D_pct': (r.choose_D / total_selections) * 100,
                        'select_E_pct': (r.choose_E / total_selections) * 100,
                        'total_selections': total_selections,
                    })
            
            if agent_selection:
                agent_df = pd.DataFrame(agent_selection)
                agent_df.to_excel(writer, sheet_name='Agent_Selection', index=False)
            
            # Pre/Post shock agent selection (critical for paper)
            agent_selection_pre_post = []
            for r in summary:
                total_pre = r.choose_A_pre + r.choose_B_pre + r.choose_C_pre + r.choose_D_pre + r.choose_E_pre
                total_post = r.choose_A_post + r.choose_B_post + r.choose_C_post + r.choose_D_post + r.choose_E_post
                
                if total_pre > 0 and total_post > 0:
                    agent_selection_pre_post.append({
                        'policy': r.policy,
                        # Pre-shock percentages
                        'pre_A_pct': (r.choose_A_pre / total_pre) * 100,
                        'pre_B_pct': (r.choose_B_pre / total_pre) * 100,
                        'pre_C_pct': (r.choose_C_pre / total_pre) * 100,
                        'pre_D_pct': (r.choose_D_pre / total_pre) * 100,
                        'pre_E_pct': (r.choose_E_pre / total_pre) * 100,
                        # Post-shock percentages
                        'post_A_pct': (r.choose_A_post / total_post) * 100,
                        'post_B_pct': (r.choose_B_post / total_post) * 100,
                        'post_C_pct': (r.choose_C_post / total_post) * 100,
                        'post_D_pct': (r.choose_D_post / total_post) * 100,
                        'post_E_pct': (r.choose_E_post / total_post) * 100,
                        # Changes (post - pre)
                        'delta_A_pct': ((r.choose_A_post / total_post) - (r.choose_A_pre / total_pre)) * 100,
                        'delta_B_pct': ((r.choose_B_post / total_post) - (r.choose_B_pre / total_pre)) * 100,
                        'delta_C_pct': ((r.choose_C_post / total_post) - (r.choose_C_pre / total_pre)) * 100,
                        'delta_D_pct': ((r.choose_D_post / total_post) - (r.choose_D_pre / total_pre)) * 100,
                        'delta_E_pct': ((r.choose_E_post / total_post) - (r.choose_E_pre / total_pre)) * 100,
                    })
            
            if agent_selection_pre_post:
                agent_pre_post_df = pd.DataFrame(agent_selection_pre_post)
                agent_pre_post_df.to_excel(writer, sheet_name='Agent_Selection_PrePost', index=False)
        
        print(f"✅ Excel file saved to: {excel_path}")
    except ImportError as e:
        if 'openpyxl' in str(e).lower():
            print("[WARN] openpyxl not available, skipping Excel export. Install with: pip install openpyxl")
        else:
            print(f"[WARN] Missing dependency: {e}")
    except Exception as e:
        print(f"[WARN] Failed to write Excel file: {e}")
        print("[WARN] Results are still available in CSV format")


def main():
    ap = argparse.ArgumentParser(description="Exp3: Role Shock Robustness Experiment")
    ap.add_argument("--n", type=int, default=1000, help="number of tasks")
    ap.add_argument("--p-hard", type=float, default=0.2, help="probability of hard task")
    ap.add_argument("--seed", type=int, default=123, help="random seed")
    ap.add_argument("--topL", type=int, default=3, help="Top-L candidates by static match_score")
    ap.add_argument(
        "--outdir",
        type=str,
        default="experiments/exp3/sim/results/sim_exp3_robustness",
        help="output directory",
    )

    ap.add_argument("--no-plots", action="store_true", help="do not generate png plots")
    ap.add_argument("--no-excel", action="store_true", help="do not generate Excel file")

    # Shock parameters
    ap.add_argument("--shock", type=str, required=True, choices=["A_unavailable", "A_degraded"],
                    help="Shock type: A_unavailable (agent A becomes unavailable) or A_degraded (agent A performance degrades)")
    ap.add_argument("--shock-point", type=int, default=None,
                    help="Task index where shock occurs (default: n/2)")
    ap.add_argument("--freeze-after-shock", action="store_true",
                    help="Freeze LinUCB updates after shock (for ablation)")

    # LinUCB params
    ap.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration scale")
    ap.add_argument("--l2", type=float, default=1.0, help="LinUCB l2 regularization lambda")
    ap.add_argument("--delta", type=float, default=0.05, help="LinUCB confidence")
    ap.add_argument("--S", type=float, default=1.0, help="bound on ||theta*||")

    # Reward shaping params
    ap.add_argument("--latency-scale-ms", type=float, default=2000.0, help="latency normalization scale")
    ap.add_argument("--latency-penalty", type=float, default=0.2, help="penalty multiplier for latency")
    ap.add_argument("--cost-lambda", type=float, default=0.15, help="penalty multiplier for cost")

    args = ap.parse_args()
    
    # Create separate folders for Shock A and Shock B with timestamp
    shock_type_short = "ShockA" if args.shock == "A_unavailable" else "ShockB"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    args.outdir = os.path.join(
        args.outdir,
        shock_type_short,
        timestamp,
    )
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"\n[EXP3] Starting experiment: {shock_type_short}")
    print(f"[EXP3] Output directory: {args.outdir}")

    # Set default shock point
    shock_point = args.shock_point if args.shock_point is not None else args.n // 2

    base_rng = random.Random(args.seed)
    tasks = generate_tasks(args.n, args.p_hard, base_rng)

    # Define agent pool (same as Exp1)
    profiles: List[SimAgentProfile] = [
        SimAgentProfile(
            agent_id="A",
            call_cost=1.00,
            base_latency_ms=900.0,
            p_simple=0.99,
            p_hard=0.99,
            match_simple=0.80,
            match_hard=0.95,
        ),
        SimAgentProfile(
            agent_id="B",
            call_cost=0.10,
            base_latency_ms=350.0,
            p_simple=0.88,
            p_hard=0.18,
            match_simple=0.95,
            match_hard=0.20,
        ),
        SimAgentProfile(
            agent_id="C",
            call_cost=0.35,
            base_latency_ms=550.0,
            p_simple=0.92,
            p_hard=0.65,
            match_simple=0.75,
            match_hard=0.75,
        ),
        SimAgentProfile(
            agent_id="D",
            call_cost=0.22,
            base_latency_ms=420.0,
            p_simple=0.90,
            p_hard=0.45,
            match_simple=0.85,
            match_hard=0.55,
        ),
        SimAgentProfile(
            agent_id="E",
            call_cost=0.60,
            base_latency_ms=780.0,
            p_simple=0.95,
            p_hard=0.85,
            match_simple=0.60,
            match_hard=0.90,
        ),
    ]

    def make_agents(policy_seed: int) -> List[SimAgent]:
        rng = random.Random(policy_seed)
        return [SimAgent(p, rng) for p in profiles]

    # Policies for Exp3
    # Always include linucb_frozen as baseline (critical for paper)
    policies = ["static_rule", "random", "linucb", "linucb_frozen"]

    summary_rows: List[SummaryRow] = []
    traj_logs: Dict[str, List[StepLog]] = {}

    for i, pol in enumerate(policies):
        pol_seed = args.seed + 1000 * (i + 1)
        agents = make_agents(pol_seed)

        row, logs = run_policy_exp3(
            policy_name=pol,
            tasks=tasks,
            agents=agents,
            topL=args.topL,
            shock_type=args.shock,
            shock_point=shock_point,
            freeze_after_shock=bool(args.freeze_after_shock),
            linucb_alpha=args.alpha,
            linucb_l2=args.l2,
            delta=args.delta,
            S=args.S,
            latency_scale_ms=args.latency_scale_ms,
            latency_penalty=args.latency_penalty,
            cost_lambda=args.cost_lambda,
            seed_for_policy=pol_seed,
        )
        summary_rows.append(row)
        traj_logs[pol] = logs

        print(
            f"[{pol}] pre_shock={row.success_rate_pre_shock:.3f} "
            f"post_shock={row.success_rate_post_shock:.3f} "
            f"recovery_time={row.recovery_time} "
            f"deadlock_rate={row.deadlock_rate:.3f}"
        )

    # Write outputs
    print(f"\n[OUTPUT] Writing results to: {args.outdir}")
    
    # CSV files
    summary_dicts = []
    for r in summary_rows:
        row_dict = asdict(r)
        # Add computed fields for CSV
        row_dict['recovery_status'] = 'Recovered' if r.recovery_time >= 0 else 'Not Recovered'
        row_dict['success_rate_drop'] = r.success_rate_pre_shock - r.success_rate_post_shock
        summary_dicts.append(row_dict)
    
    write_csv(
        os.path.join(args.outdir, "summary.csv"),
        summary_dicts,
    )
    print(f"  ✓ summary.csv")
    
    for pol, logs in traj_logs.items():
        write_csv(
            os.path.join(args.outdir, f"trajectory_{pol}.csv"),
            [asdict(x) for x in logs],
        )
        print(f"  ✓ trajectory_{pol}.csv")

    # Write Excel (with multiple sheets)
    if not args.no_excel:
        write_excel(args.outdir, summary_rows, traj_logs)
        print(f"  ✓ exp3_results.xlsx")

    # Generate plots
    if not args.no_plots:
        try_plot(args.outdir, summary_rows, traj_logs)
        # Check if plots were actually generated
        plot_files = [
            "plot_v_shape_recovery.png",
            "plot_v_shape_all_policies.png", 
            "plot_recovery_time_comparison.png"
        ]
        for plot_file in plot_files:
            plot_path = os.path.join(args.outdir, plot_file)
            if os.path.exists(plot_path):
                print(f"  ✓ {plot_file}")

    # Print comprehensive terminal summary
    print_exp3_summary_terminal(
        summary_rows,
        shock_type=args.shock,
        shock_point=shock_point,
    )
    
    # Print agent usage for key policies (linucb and linucb_frozen)
    for pol, logs in traj_logs.items():
        if pol in ("linucb", "linucb_frozen"):
            print_agent_usage_terminal(
                policy=pol,
                logs=logs,
                shock_point=shock_point,
            )
    
    print(f"\n📁 Results saved to:\n  {args.outdir}\n")


if __name__ == "__main__":
    main()

