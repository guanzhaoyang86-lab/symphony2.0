"""
Symphony 2.0 core execution engine (Path-1: centralized orchestrator).

✅ Symphony 2.0 features:
- Two-stage agent selection:
    Stage-1: Top-L by static capability match_score (Symphony 1.0 compatible)
    Stage-2: Global LinUCB selects within Top-L using dynamic_state features
- Multi-CoT execution and voting per subtask
- Online update after voting (winner bonus + latency penalty)
- Optional Symphony 1.0-style planning decomposition (multiple planners produce chains)

✅ This patched version additionally:
- Planner branch also performs online updates (closes the loop in planner mode)
- Planner weighted vote keys on extracted Final (not whole raw text)
- Never uses x[1] as match_score (build_x may normalize -> x[1] != raw match)
- Safer fallback: do not pick unavailable agent when pool is empty
- Optional correctness reward if gold label is provided in task/subtask context

Return modes:
- "aggregate": returns a multi-subtask report (string)
- "final": returns final answer only (string, BBH-friendly)
- "trace": returns dict with per-run traces (for debugging / saving)
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------- imports (package / local) ----------------------
try:
    # Package mode
    from symphony.protocol.task_contract import Task  # type: ignore
    from symphony.agents.agent import Agent  # type: ignore
    from symphony.core.linucb_selector import GlobalLinUCB, build_x  # type: ignore
except Exception:  # pragma: no cover
    # Local mode
    from protocol.task_contract import Task  # type: ignore
    from agents.agent import Agent  # type: ignore
    from core.linucb_selector import GlobalLinUCB, build_x  # type: ignore

# Risk guard is optional
try:
    from core.risk_guard import RiskAwareGuard, RiskGuardConfig  # type: ignore
except Exception:  # pragma: no cover
    RiskAwareGuard = None  # type: ignore
    RiskGuardConfig = None  # type: ignore


# ---------------------- orchestrator ----------------------
class SymphonyOrchestrator:
    """Main orchestrator for multi-agent task execution (Path-1)."""

    def __init__(
            self,
            verbose: bool = False,
            # ---- Dynamic Beacon Selection knobs (2.0) ----
            use_dynamic: bool = True,
            topL: int = 3,
            linucb_alpha: float = 1.0,
            linucb_l2: float = 1.0,
            latency_scale_ms: float = 2000.0,
            latency_penalty: float = 0.2,
            win_bonus: float = 0.5,
            # ---- Optional correctness reward (for better scores) ----
            correctness_bonus: float = 0.0,  # add when voted final matches gold
            incorrect_penalty: float = 0.0,  # subtract when voted final mismatches gold
            # ---- Optional Symphony 1.0 planner ----
            plan_k: int = 1,
            # ---- Risk Guard ----
            enable_risk_guard: bool = False,
    ) -> None:
        self.lock = threading.Lock()
        self.verbose = bool(verbose)

        # agent registry
        self.agents: List[Agent] = []

        # dynamic selection knobs
        self.use_dynamic = bool(use_dynamic)
        self.topL = max(1, int(topL))
        self.latency_scale_ms = float(latency_scale_ms)
        self.latency_penalty = float(latency_penalty)
        self.win_bonus = float(win_bonus)

        # correctness reward knobs
        self.correctness_bonus = float(correctness_bonus)
        self.incorrect_penalty = float(incorrect_penalty)

        # planner knobs (Symphony 1.0 style)
        self.plan_k = max(1, int(plan_k))

        # optional risk guard
        self.enable_risk_guard = bool(enable_risk_guard)
        self.risk_guard = None
        if self.enable_risk_guard and RiskAwareGuard is not None and RiskGuardConfig is not None:
            self.risk_guard = RiskAwareGuard(RiskGuardConfig())

        # ✅ Global LinUCB (single global A,b)
        # build_x returns a vector (commonly 6-dim): [1, match, load, lat_norm, rep, available]
        self.selector: Optional[GlobalLinUCB] = None
        if self.use_dynamic:
            self.selector = GlobalLinUCB(d=6, l2=float(linucb_l2), alpha=float(linucb_alpha))

    # ---------------------- logging ----------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---------------------- lifecycle ----------------------
    def register_agent(self, agent: Agent) -> None:
        with self.lock:
            if agent not in self.agents:
                self.agents.append(agent)
                self._log(f"[ORCHESTRATOR] Registered agent: {getattr(agent, 'agent_id', '')}")

    def get_registered_agents(self) -> List[Agent]:
        return list(self.agents)

    # ---------------------- helpers: requirement normalization ----------------------
    @staticmethod
    def _normalize_requirement(req: str) -> str:
        """
        Make matching slightly more robust:
        - lowercase
        - spaces/hyphens -> underscores
        """
        r = (req or "").strip().lower()
        r = re.sub(r"[\s\-]+", "_", r)
        return r

    # ---------------------- helpers: gold/correctness ----------------------
    def _get_gold_from_context(self, ctx: Dict[str, Any]) -> Union[None, str, List[str]]:
        """
        Look for gold label in context. Supported:
          ctx["gold"] = "yes" or "A" or "42"
          ctx["gold"] = ["A", "B"]  (multiple acceptable)
        """
        if not isinstance(ctx, dict):
            return None
        gold = ctx.get("gold", None)
        if gold is None:
            return None
        if isinstance(gold, str):
            g = gold.strip()
            return g if g else None
        if isinstance(gold, (list, tuple, set)):
            out: List[str] = []
            for x in gold:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out if out else None
        return None

    def _canon_answer(self, s: str) -> str:
        """
        Canonicalize an answer for comparison:
        - try extract_final first
        - strip, collapse spaces, lowercase for yes/no/valid/invalid, keep letter uppercase
        """
        t = (s or "").strip()
        fin = self._extract_final_from_text(t)
        t = (fin or t).strip()
        t = re.sub(r"\s+", " ", t)

        # normalize common BBH labels
        low = t.lower()
        if low in ("yes", "no", "valid", "invalid", "true", "false"):
            return low
        if re.fullmatch(r"[A-Za-z]", t):
            return t.upper()
        return t

    def _is_correct(self, pred_text: str, gold: Union[str, List[str], None]) -> Optional[bool]:
        if gold is None:
            return None
        pred = self._canon_answer(pred_text)
        if isinstance(gold, str):
            return pred == self._canon_answer(gold)
        if isinstance(gold, list):
            gold_set = {self._canon_answer(x) for x in gold if isinstance(x, str)}
            return pred in gold_set if gold_set else None
        return None

    # ---------------------- public entry ----------------------
    def execute_task(
            self,
            task: Task,
            cot_count: int = 3,
            return_mode: str = "aggregate",  # "aggregate" | "final" | "trace"
    ) -> Any:
        """
        Main execution:
        - If plan_k > 1: planning => execute each plan chain => weighted vote.
        - Else: decompose by task.requirements => execute each subtask with multi-CoT voting.
        """
        task_text = getattr(task, "description", "") or ""
        ctx = getattr(task, "context", {}) or {}

        if not self.agents:
            if return_mode == "trace":
                return {"error": "No agents registered", "results": {}, "traces": {}}
            return "[ERROR] No agents registered"

        # ---------- (A) Optional planner mode (Symphony 1.0-style, patched to 2.0 loop) ----------
        if self.plan_k > 1:
            plans = self._plan_chains_v1(task_text=task_text, ctx=ctx, m=self.plan_k)
            plan_answers: List[str] = []
            plan_weights: List[float] = []
            plan_traces: List[Dict[str, Any]] = []

            for p in plans:
                ans, w, tr = self._run_plan_chain_v1(base_task=task_text, chain=p["chain"], base_ctx=ctx)
                plan_answers.append(ans)
                plan_weights.append(w)
                plan_traces.append({"planner": p.get("planner", ""), "w": w, "trace": tr, "chain": p["chain"]})

            # ✅ vote on extracted final keys (not whole raw text)
            plan_keys = [(self._extract_final_from_text(a) or str(a).strip()) for a in plan_answers]
            win_key = self._weighted_vote(plan_keys, plan_weights)

            # choose representative full text with max weight among those sharing win_key
            best_i = 0
            best_w = -1e18
            for i, (k, w) in enumerate(zip(plan_keys, plan_weights)):
                if k == win_key and float(w) > best_w:
                    best_w = float(w)
                    best_i = i
            final_text = plan_answers[best_i] if plan_answers else ""

            # ✅ optional correctness reward at planner-level
            gold = self._get_gold_from_context(ctx)
            correct = self._is_correct(win_key, gold)

            # ✅ winner-bonus updates for all steps in winning trajectory(ies)
            if self.use_dynamic and self.selector is not None:
                for i, k in enumerate(plan_keys):
                    if k != win_key:
                        continue
                    tr = plan_traces[i].get("trace", {}) if isinstance(plan_traces[i], dict) else {}
                    recs = tr.get("records", [])
                    for rec in recs:
                        x = rec.get("x")
                        if isinstance(x, list):
                            bonus = float(self.win_bonus)
                            if correct is True:
                                bonus += float(self.correctness_bonus)
                            elif correct is False:
                                bonus -= float(self.incorrect_penalty)
                            self.selector.update(x, bonus)

            if return_mode == "trace":
                return {
                    "final": win_key,
                    "final_text": final_text,
                    "answers": plan_answers,
                    "keys": plan_keys,
                    "weights": plan_weights,
                    "plans": plan_traces,
                    "gold": gold,
                    "correct": correct,
                }

            if return_mode == "final":
                return (win_key or (self._extract_final_from_text(final_text) or final_text)).strip()

            # aggregate
            rep = "## Symphony Planner Result\n\n"
            rep += f"**Original Task**: {task_text}\n\n"
            for i, (a, w) in enumerate(zip(plan_answers, plan_weights), 1):
                rep += f"{i}. (w={w:.3f}) {a}\n\n"
            rep += f"\n**Final answer**: {win_key}\n"
            return rep.strip()

        # ---------- (B) Default non-planner mode ----------
        subtasks = self._decompose_task(task)
        if not subtasks:
            subtasks = [self._mk_subtask(task_text, ctx, i=1, requirement="general-reasoning")]

        # normalize
        for i, st in enumerate(subtasks):
            if not isinstance(st, dict):
                st = {"input": str(st)}
                subtasks[i] = st
            st.setdefault("id", f"sub_{i + 1}")
            st.setdefault("requirement", "general-reasoning")
            st.setdefault("context", ctx)
            st.setdefault("original_task", task_text)
            st.setdefault("description", st.get("input") or st.get("description") or task_text)
            if not st.get("input"):
                st["input"] = st.get("description") or task_text

            # normalize req for matching
            st["requirement"] = self._normalize_requirement(str(st.get("requirement", "general-reasoning")))

        agent_assignments = self._find_suitable_agents(subtasks)

        out = self._execute_with_cot_voting(
            subtasks=subtasks,
            agent_assignments=agent_assignments,
            cot_count=cot_count,
            return_mode=return_mode,
        )

        if return_mode == "trace":
            if isinstance(out, dict) and "results" in out and isinstance(out["results"], dict):
                if len(out["results"]) == 1:
                    one = next(iter(out["results"].values()))
                    out["final"] = self._extract_final_from_text(str(one)) or str(one)
            return out

        if return_mode == "final":
            if isinstance(out, dict) and len(out) == 1:
                s = str(next(iter(out.values()))).strip()
                s = re.sub(r"(?is)</?\s*answer\s*>", "", s).strip()
                fin = self._extract_final_from_text(s)
                return (fin or s).strip()

            aggregated = self._aggregate_results(out, task)  # type: ignore[arg-type]
            fin = self._extract_final_from_text(aggregated)
            return fin.strip() if fin else aggregated.strip()

        return self._aggregate_results(out, task)  # type: ignore[arg-type]

    # ---------------------- decomposition (simple baseline) ----------------------
    def _mk_subtask(self, task_text: str, ctx: Dict[str, Any], i: int, requirement: str) -> Dict[str, Any]:
        return {
            "id": f"{uuid.uuid4().hex}_sub_{i}",
            "requirement": self._normalize_requirement(requirement),
            "input": task_text,
            "description": task_text,
            "context": ctx or {},
            "original_task": task_text,
        }

    def _decompose_task(self, task: Task) -> List[Dict[str, Any]]:
        """
        Baseline decomposition: one subtask per requirement.
        """
        reqs = list(getattr(task, "requirements", []) or [])
        if not reqs:
            reqs = ["general-reasoning"]

        out: List[Dict[str, Any]] = []
        for i, r in enumerate(reqs, 1):
            out.append(
                self._mk_subtask(
                    task_text=getattr(task, "description", "") or "",
                    ctx=getattr(task, "context", {}) or {},
                    i=i,
                    requirement=str(r),
                )
            )
        return out

    # ---------------------- agent matching ----------------------
    def _agent_state(self, agent: Agent) -> Dict[str, Any]:
        """Symphony 2.0: read dynamic state if provided; else fallback defaults."""
        if hasattr(agent, "get_dynamic_state"):
            try:
                st = agent.get_dynamic_state()  # type: ignore[attr-defined]
                if isinstance(st, dict):
                    st.setdefault("available", True)
                    st.setdefault("load", 0.0)
                    st.setdefault("latency_ms", 500.0)
                    st.setdefault("reputation", 0.5)
                    return st
            except Exception:
                pass
        return {"available": True, "load": 0.0, "latency_ms": 500.0, "reputation": 0.5}

    def _find_suitable_agents(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return candidates per subtask:
          [{"agent": Agent, "match_score": float}, ...]
        """
        assignments: Dict[str, List[Dict[str, Any]]] = {}
        for st in subtasks:
            sid = st["id"]
            req = self._normalize_requirement(str(st.get("requirement", "general-reasoning")))
            cand: List[Dict[str, Any]] = []

            for ag in self.agents:
                ms = 0.0
                if hasattr(ag, "capability_manager"):
                    try:
                        ms = float(ag.capability_manager.match(req))  # type: ignore[attr-defined]
                    except Exception:
                        ms = 0.0
                cand.append({"agent": ag, "match_score": ms})

            cand.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
            assignments[sid] = cand
        return assignments

    def _select_agent_dynamic(
            self,
            candidates: List[Dict[str, Any]],
            used_ids: set,
    ) -> Tuple[Agent, List[float], Dict[str, Any], float]:
        """
        Stage-1: Top-L by match_score (already sorted)
        Stage-2: Global LinUCB selects within Top-L using build_x(match, dynamic_state, available)

        Return: (agent, x, state, raw_match_score)
        """
        topL = candidates[: self.topL]

        pool: List[Tuple[str, List[float], Agent, Dict[str, Any], float]] = []
        for c in topL:
            agent = c["agent"]
            aid = str(getattr(agent, "agent_id", ""))

            if aid in used_ids and len(used_ids) < len(topL):
                continue

            st = self._agent_state(agent)
            if not bool(st.get("available", True)):
                continue

            raw_ms = float(c.get("match_score", 0.0))
            x = build_x(
                match_score=raw_ms,
                dynamic_state={
                    "load": float(st.get("load", 0.0)),
                    "latency_ms": float(st.get("latency_ms", 500.0)),
                    "reputation": float(st.get("reputation", 0.5)),
                },
                available=bool(st.get("available", True)),
                latency_scale_ms=float(self.latency_scale_ms),
            )
            pool.append((aid, x, agent, st, raw_ms))

        # ✅ safer fallback: pick first AVAILABLE in topL; else pick topL[0] but mark available=False in x
        if not pool:
            for c in topL:
                ag = c["agent"]
                st0 = self._agent_state(ag)
                if bool(st0.get("available", True)):
                    raw_ms0 = float(c.get("match_score", 0.0))
                    x0 = build_x(
                        match_score=raw_ms0,
                        dynamic_state={
                            "load": float(st0.get("load", 0.0)),
                            "latency_ms": float(st0.get("latency_ms", 500.0)),
                            "reputation": float(st0.get("reputation", 0.5)),
                        },
                        available=True,
                        latency_scale_ms=float(self.latency_scale_ms),
                    )
                    return ag, x0, st0, raw_ms0

            c0 = topL[0]
            ag0 = c0["agent"]
            st0 = self._agent_state(ag0)
            raw_ms0 = float(c0.get("match_score", 0.0))
            x0 = build_x(
                match_score=raw_ms0,
                dynamic_state={
                    "load": float(st0.get("load", 0.0)),
                    "latency_ms": float(st0.get("latency_ms", 500.0)),
                    "reputation": float(st0.get("reputation", 0.5)),
                },
                available=False,
                latency_scale_ms=float(self.latency_scale_ms),
            )
            return ag0, x0, st0, raw_ms0

        assert self.selector is not None
        chosen_id = self.selector.select([(aid, x) for (aid, x, _, _, _) in pool])
        for (aid, x, agent, st, raw_ms) in pool:
            if aid == chosen_id:
                return agent, x, st, raw_ms
        return pool[0][2], pool[0][1], pool[0][3], pool[0][4]

    # ---------------------- core: multi-CoT + voting (+ trace) ----------------------
    def _execute_with_cot_voting(
            self,
            subtasks: List[Dict[str, Any]],
            agent_assignments: Dict[str, List[Dict[str, Any]]],
            cot_count: int,
            return_mode: str = "final",  # "final" | "aggregate" | "trace"
    ) -> Any:
        """
        For each subtask:
        - run up to `cot_count` times (bounded by |TopL|)
        - vote among outputs (keys on extracted Final)
        - update LinUCB online after vote (winner bonus + latency penalty + optional correctness reward)
        """
        results: Dict[str, str] = {}
        traces_by_subtask: Dict[str, Any] = {}

        for st in subtasks:
            sid = st["id"]
            req = str(st.get("requirement", "general-reasoning"))
            candidates = agent_assignments.get(sid, [])

            if not candidates:
                err = f"[ERROR] No agents available for subtask: {req}"
                results[sid] = err
                if return_mode == "trace":
                    traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                continue

            # filter available
            candidates_avail: List[Dict[str, Any]] = []
            for c in candidates:
                ag = c["agent"]
                stt = self._agent_state(ag)
                if bool(stt.get("available", True)):
                    candidates_avail.append(c)

            if not candidates_avail:
                err = f"[ERROR] No AVAILABLE agents for subtask: {req}"
                results[sid] = err
                if return_mode == "trace":
                    traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                continue

            topL = candidates_avail[: max(1, self.topL)]
            runs = min(int(cot_count), len(topL)) if topL else 0
            if runs <= 0:
                err = f"[ERROR] All agents filtered out for subtask: {req}"
                results[sid] = err
                if return_mode == "trace":
                    traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                continue

            used_ids = set()
            run_records: List[Dict[str, Any]] = []
            cot_results: List[str] = []

            for _ in range(runs):
                if self.use_dynamic and self.selector is not None:
                    agent, x, _st, match_score = self._select_agent_dynamic(topL, used_ids)
                else:
                    agent = topL[0]["agent"]
                    stt = self._agent_state(agent)
                    match_score = float(topL[0].get("match_score", 0.0))
                    x = build_x(
                        match_score=match_score,
                        dynamic_state={
                            "load": float(stt.get("load", 0.0)),
                            "latency_ms": float(stt.get("latency_ms", 500.0)),
                            "reputation": float(stt.get("reputation", 0.5)),
                        },
                        available=bool(stt.get("available", True)),
                        latency_scale_ms=float(self.latency_scale_ms),
                    )

                aid = str(getattr(agent, "agent_id", ""))
                used_ids.add(aid)

                t0 = time.time()
                try:
                    text = self._execute_subtask_on_agent(agent, st)
                except Exception as e:
                    text = f"[AGENT_ERROR] {str(e)}"
                dt_ms = (time.time() - t0) * 1000.0

                cot_results.append(text)
                run_records.append(
                    {
                        "agent_id": aid,
                        "match_score": float(match_score),
                        "x": x,
                        "latency_ms": float(dt_ms),
                        "text": text,
                        "final": self._extract_final_from_text(text) or "",
                    }
                )

            final_result = self._vote_on_results(cot_results, st)

            # optional risk guard
            if self.enable_risk_guard and self.risk_guard is not None:
                decision = self.risk_guard.assess(
                    task_context=st.get("context", {}) or {},
                    subtask=st,
                    cot_results=cot_results,
                    voted_text=final_result,
                    history_risk=0.0,
                )
                if getattr(decision, "action", None) == "rerun":
                    extra = int(getattr(getattr(self.risk_guard, "cfg", None), "extra_cot", 1))
                    for _ in range(extra):
                        if self.use_dynamic and self.selector is not None:
                            agent, x, _st, match_score = self._select_agent_dynamic(topL, used_ids)
                        else:
                            agent = topL[0]["agent"]
                            stt = self._agent_state(agent)
                            match_score = float(topL[0].get("match_score", 0.0))
                            x = build_x(
                                match_score=match_score,
                                dynamic_state={
                                    "load": float(stt.get("load", 0.0)),
                                    "latency_ms": float(stt.get("latency_ms", 500.0)),
                                    "reputation": float(stt.get("reputation", 0.5)),
                                },
                                available=bool(stt.get("available", True)),
                                latency_scale_ms=float(self.latency_scale_ms),
                            )

                        aid = str(getattr(agent, "agent_id", ""))
                        used_ids.add(aid)
                        t0 = time.time()
                        try:
                            text = self._execute_subtask_on_agent(agent, st)
                        except Exception as e:
                            text = f"[AGENT_ERROR] {str(e)}"
                        dt_ms = (time.time() - t0) * 1000.0

                        cot_results.append(text)
                        run_records.append(
                            {
                                "agent_id": aid,
                                "match_score": float(match_score),
                                "x": x,
                                "latency_ms": float(dt_ms),
                                "text": text,
                                "final": self._extract_final_from_text(text) or "",
                            }
                        )
                    final_result = self._vote_on_results(cot_results, st)

            results[sid] = final_result

            if self.use_dynamic and self.selector is not None:
                self._online_update_after_vote(run_records, final_result, st)

            finals = []
            for r in run_records:
                f = (r.get("final") or "").strip()
                finals.append(f)

            vote_count = Counter(finals)

            vote_weight = defaultdict(float)
            for r in run_records:
                f = (r.get("final") or "").strip()
                vote_weight[f] += float(r.get("match_score", 0.0))  # 用 match_score 做一个“加权票”观测

            if return_mode == "trace":
                ctx = st.get("context", {}) or {}
                gold = self._get_gold_from_context(ctx)
                correct = self._is_correct(final_result, gold)
                traces_by_subtask[sid] = {
                    "requirement": req,
                    "context": ctx,
                    "gold": gold,
                    "vote_count": dict(vote_count),
                    "vote_weight_by_match_score": dict(vote_weight),
                    "correct": correct,
                    "runs": run_records,
                    "voted": final_result,
                    "voted_final": self._extract_final_from_text(final_result) or "",
                }

        if return_mode == "trace":
            return {"results": results, "traces": traces_by_subtask}
        return results

    def _online_update_after_vote(self, run_records, voted_text, subtask) -> None:
        voted_final = self._extract_final_from_text(voted_text) or voted_text.strip()

        ctx = subtask.get("context", {}) or {}
        gold = self._get_gold_from_context(ctx)
        correct = self._is_correct(voted_final, gold)

        for rec in run_records:
            x = rec.get("x")
            if not isinstance(x, list):
                continue

            latency_ms = float(rec.get("latency_ms", 0.0))
            text = str(rec.get("text", ""))

            ok = (text.strip() != "") and (not text.startswith("[ERROR]")) and (not text.startswith("[AGENT_ERROR]"))
            agent_final = self._extract_final_from_text(text) or text.strip()
            is_winner = (agent_final == voted_final) and (voted_final != "")

            lat_norm = min(1.0, latency_ms / max(1.0, self.latency_scale_ms))
            penalty = self.latency_penalty * (lat_norm ** 0.5)

            corr_term = 0.0
            if correct is True:
                corr_term += float(self.correctness_bonus)
            elif correct is False:
                corr_term -= float(self.incorrect_penalty)

            win_term = (self.win_bonus if is_winner else 0.0)
            base_term = (1.0 if ok else 0.0)

            reward = base_term + win_term + corr_term - penalty

            # ✅ 关键：把“每轮得分”写回 trace
            rec["vote_voted_final"] = voted_final
            rec["vote_ok"] = ok
            rec["vote_is_winner"] = is_winner
            rec["vote_base_term"] = base_term
            rec["vote_win_term"] = win_term
            rec["vote_corr_term"] = corr_term
            rec["vote_latency_penalty"] = penalty
            rec["vote_reward"] = reward

            self.selector.update(x, reward)

    # ---------------------- calling an agent safely ----------------------
    def _execute_subtask_on_agent(self, agent: Agent, subtask: Dict[str, Any]) -> str:
        """
        Compatibility shim:
        build legacy agent_task and call agent.execute_task(...).
        """
        instruction = (
                subtask.get("input")
                or subtask.get("description")
                or subtask.get("original_task")
                or ""
        )
        requirement = str(subtask.get("requirement", "general-reasoning"))
        original_problem = subtask.get("original_task", instruction)

        agent_task = {
            "subtask_id": 1,
            "steps": {"1": [instruction, requirement]},
            "previous_results": subtask.get("previous_results", []) or [],
            "original_problem": original_problem,
            "final_result": "",
            "user_id": "symphony_orchestrator",
        }

        # no-model simulation path
        if hasattr(agent, "base_model") and getattr(agent, "base_model") is None:
            domain = (subtask.get("context", {}) or {}).get("domain", "general")
            return f"[SIMULATED] Agent {getattr(agent, 'agent_id', '')} handled {requirement} ({domain})."

        if not hasattr(agent, "execute_task"):
            return f"[ERROR] Agent {getattr(agent, 'agent_id', '')} has no execute_task()"

        result = agent.execute_task(agent_task)  # type: ignore[call-arg]

        # --- unwrap common return formats ---
        if result is None:
            return ""

        # avoid numeric/status-code becoming "1"
        if isinstance(result, (int, float, bool)):
            return ""

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            # OpenAI / OpenAI-compatible
            if "choices" in result:
                try:
                    c0 = (result.get("choices") or [{}])[0]
                    if isinstance(c0, dict):
                        msg = c0.get("message") or {}
                        if isinstance(msg, dict) and msg.get("content"):
                            return str(msg["content"]).strip()
                        if c0.get("text"):
                            return str(c0["text"]).strip()
                except Exception:
                    pass

            for k in ("final_result", "final_text", "text", "answer", "output", "content"):
                if k in result and result[k]:
                    return str(result[k]).strip()

            # Symphony might return {"<uuid>_sub_1": "<text>"}
            for v in result.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()

            return ""

        # object-like
        for attr in ("final_result", "final_text", "text", "answer", "output", "content"):
            if hasattr(result, attr):
                val = getattr(result, attr, None)
                if val:
                    return str(val).strip()

        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict):
                    for k in ("final_result", "final_text", "text", "answer", "output", "content"):
                        if k in d and d[k]:
                            return str(d[k]).strip()
            except Exception:
                pass

        return str(result).strip()

    # ---------------------- voting ----------------------
    def _vote_on_results(self, cot_results: List[str], subtask: Dict[str, Any]) -> str:
        if len(cot_results) == 1:
            return cot_results[0]

        finals: Dict[str, int] = {}
        for r in cot_results:
            fx = self._extract_final_from_text(r)
            if fx:
                finals[fx] = finals.get(fx, 0) + 1

        if finals:
            best_ans = max(finals.items(), key=lambda kv: kv[1])[0]
            tied = [r for r in cot_results if (self._extract_final_from_text(r) == best_ans)]
            if tied:
                return max(tied, key=len)

        valid_results = [r for r in cot_results if not r.startswith("[ERROR]") and not r.startswith("[AGENT_ERROR]")]
        return max(valid_results, key=len) if valid_results else cot_results[0]

    def _weighted_vote(self, answers: List[str], weights: List[float]) -> str:
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]
        score: Dict[str, float] = {}
        for a, w in zip(answers, weights):
            key = (a or "").strip()
            score[key] = score.get(key, 0.0) + float(w)
        return max(score.items(), key=lambda kv: kv[1])[0] if score else answers[0]

    # ---------------------- robust final extractor (BBH-friendly) ----------------------
    @staticmethod
    def _extract_final_from_text(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None

        # 1) boxed
        m = re.findall(r"\\boxed\{([^}]*)\}", s)
        if m:
            return m[-1].strip()

        # 2) explicit "Final answer:" / "答案："
        m2 = re.search(r"(?im)^\s*(?:final\s*answer|answer|答案)\s*[:：]\s*([^\n\r]+)\s*$", s)
        if m2:
            cand = m2.group(1).strip()

            # ✅ dyck: prioritize bracket-only substring BEFORE any split on '['
            cand_ns = re.sub(r"\s+", "", cand)
            brs = re.findall(r"[<>\[\]\(\)\{\}]+", cand_ns)
            if brs:
                return max(brs, key=len)

            # other tasks: trim trailing punctuation/paren
            cand = re.split(r"\s*(?:\.\s*|,|;|\(|（|【)", cand, maxsplit=1)[0].strip()
            if cand:
                return cand

        # 2.5) dyck: whole-line brackets
        t = s.strip()
        if re.fullmatch(r"[<>\[\]\(\)\{\}]+", t or ""):
            return t

        # 2.6) validity (formal_fallacies)
        m = re.search(r"(?i)\b(valid|invalid)\b", s)
        if m:
            return m.group(1).lower()

        # 3) True/False
        m = re.search(r"\b(true|false)\b", s, re.I)
        if m:
            return m.group(1).lower().capitalize()

        # 4) yes/no
        m = re.search(r"\b(yes|no)\b", s, re.I)
        if m:
            return m.group(1).lower()

        # 5) single choice A-Z
        m = re.search(r"\b([A-Za-z])\b(?![A-Za-z])", s)
        if m:
            return m.group(1).upper()

        # 6) numbers (last)
        m = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", s)
        if m:
            return m[-1]

        return None

    # ---------------------- aggregation ----------------------
    def _aggregate_results(self, results: Dict[str, str], original_task: Task) -> str:
        aggregated = "## Symphony Multi-Agent Task Execution Result\n\n"
        aggregated += f"**Original Task**: {getattr(original_task, 'description', '')}\n\n"
        aggregated += f"**Domain**: {getattr(original_task, 'context', {}).get('domain', 'General')}\n"
        aggregated += f"**Complexity**: {getattr(original_task, 'context', {}).get('complexity', 'Medium')}\n\n"
        aggregated += "### Subtask Results:\n\n"

        finals: List[str] = []
        for i, (sid, result) in enumerate(results.items(), 1):
            aggregated += f"{i}. **{sid}**: {result}\n\n"
            ext = self._extract_final_from_text(result)
            if ext:
                finals.append(ext)

        aggregated += (
            f"\n**Execution Summary**: Coordinated {len(results)} subtasks "
            f"via Top-L + Global LinUCB selection and CoT voting.\n"
        )

        final_ans = finals[-1] if finals else None
        if final_ans:
            aggregated += f"\n**Final answer**: {final_ans}\n"
        return aggregated.strip()

    # ---------------------- Symphony 1.0 planner (optional) ----------------------
    _PLANNER_PROMPT = """You are a problem decomposer, NOT a solver.
Break the problem into a sequence of executable subtasks.
Do NOT solve the problem.

Return STRICT JSON ONLY (no markdown), format:
{
  "subtasks": [
    "Q1: ...",
    "Q2: ...",
    ...
  ]
}

Rules:
- No final answer, no intermediate results.
- The LAST subtask should instruct producing the final answer in the required output format.
Problem:
{user_input}
"""

    def _select_planning_agents(self, k: int) -> List[Agent]:
        planners = [a for a in self.agents if "planning" in (getattr(a, "capabilities", []) or [])]
        if len(planners) >= k:
            return planners[:k]
        return list(self.agents)[:k]

    def _parse_planner_json(self, raw: str) -> Optional[List[str]]:
        try:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            s = m.group(0) if m else raw
            j = json.loads(s)
            arr = j.get("subtasks", None)
            if isinstance(arr, list) and arr:
                return [str(x) for x in arr if str(x).strip()]
        except Exception:
            return None
        return None

    def _plan_chains_v1(self, task_text: str, ctx: Dict[str, Any], m: int) -> List[Dict[str, Any]]:
        planners = self._select_planning_agents(m)
        plans: List[Dict[str, Any]] = []

        for p in planners:
            tmpl = self._PLANNER_PROMPT
            tmpl = tmpl.replace("{", "{{").replace("}", "}}").replace("{{user_input}}", "{user_input}")
            prompt = tmpl.format(user_input=task_text)

            fake_subtask = {
                "id": f"plan_{uuid.uuid4().hex}",
                "requirement": "planning",
                "input": prompt,
                "description": prompt,
                "original_task": task_text,
                "context": ctx or {},
            }
            raw = self._execute_subtask_on_agent(p, fake_subtask)
            subtasks_txt = self._parse_planner_json(raw)

            if not subtasks_txt:
                continue

            chain: List[Dict[str, Any]] = []
            for i, q in enumerate(subtasks_txt, 1):
                chain.append(
                    {
                        "id": f"{uuid.uuid4().hex}_sub_{i}",
                        "requirement": "general-reasoning",
                        "input": q,
                        "description": q,
                        "context": ctx or {},
                        "original_task": task_text,
                    }
                )
            if chain:
                plans.append({"planner": getattr(p, "agent_id", ""), "chain": chain})

        if not plans:
            plans = [{"planner": "fallback",
                      "chain": [self._mk_subtask(task_text, ctx, i=1, requirement="general-reasoning")]}]
        return plans

    def _format_executor_input(self, base_task: str, q: str, prev: List[str]) -> str:
        s = [base_task.strip()]
        if prev:
            s.append("Previous results:\n" + "\n".join(prev))
        s.append(f"Current sub-task: {q}")
        return "\n\n".join(s).strip()

    def _execute_one_subtask_beacon(self, st: Dict[str, Any], used_ids: set, base_ctx: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Planner step execution:
        - Top-L + LinUCB selection
        - Execute once per step
        - ✅ base online update here too (closes loop in planner mode)
        """
        assigns = self._find_suitable_agents([st])
        cands = assigns.get(st["id"], []) or []
        if not cands:
            return {"text": "[ERROR] No agents", "match_score": 0.0, "x": None, "latency_ms": 0.0, "agent_id": None,
                    "final": ""}

        # Prefer available within TopL
        topL = cands[: max(1, self.topL)]

        if self.use_dynamic and self.selector is not None:
            agent, x, _st, raw_ms = self._select_agent_dynamic(topL, used_ids)
            match_score = float(raw_ms)
        else:
            agent = topL[0]["agent"]
            match_score = float(topL[0].get("match_score", 0.0))
            stt = self._agent_state(agent)
            x = build_x(
                match_score=match_score,
                dynamic_state={
                    "load": float(stt.get("load", 0.0)),
                    "latency_ms": float(stt.get("latency_ms", 500.0)),
                    "reputation": float(stt.get("reputation", 0.5)),
                },
                available=bool(stt.get("available", True)),
                latency_scale_ms=float(self.latency_scale_ms),
            )

        aid = str(getattr(agent, "agent_id", ""))
        used_ids.add(aid)

        t0 = time.time()
        try:
            text = self._execute_subtask_on_agent(agent, st)
        except Exception as e:
            text = f"[AGENT_ERROR] {str(e)}"
        dt_ms = (time.time() - t0) * 1000.0

        rec = {
            "agent_id": aid,
            "match_score": float(match_score),
            "x": x,
            "latency_ms": float(dt_ms),
            "text": text,
            "final": self._extract_final_from_text(text) or "",
        }

        # ✅ planner-step base update (ok - latency penalty)
        if self.use_dynamic and self.selector is not None and isinstance(x, list):
            ok = (text.strip() != "") and (not text.startswith("[ERROR]")) and (not text.startswith("[AGENT_ERROR]"))
            lat_norm = min(1.0, dt_ms / max(1.0, self.latency_scale_ms))
            penalty = self.latency_penalty * (lat_norm ** 0.5)
            base_reward = (1.0 if ok else 0.0) - penalty
            self.selector.update(x, base_reward)

        return rec

    def _run_plan_chain_v1(self, base_task: str, chain: List[Dict[str, Any]], base_ctx: Dict[str, Any]) -> Tuple[
        str, float, Dict[str, Any]]:
        used_ids = set()
        prev: List[str] = []
        w_sum = 0.0
        steps_trace: List[Dict[str, Any]] = []
        step_records: List[Dict[str, Any]] = []

        for st in chain:
            st2 = dict(st)
            st2["input"] = self._format_executor_input(base_task=base_task, q=str(st.get("input", "")), prev=prev)

            # inherit base context (including gold if provided)
            st2_ctx = dict(base_ctx or {})
            st2_ctx.update(st2.get("context", {}) or {})
            st2["context"] = st2_ctx

            rec = self._execute_one_subtask_beacon(st2, used_ids, base_ctx=st2_ctx)
            txt = str(rec.get("text", ""))
            ms = float(rec.get("match_score", 0.0))

            prev.append(txt)
            w_sum += ms
            step_records.append(rec)
            steps_trace.append(
                {"subtask_id": st["id"], "meta": {"selected": rec.get("agent_id"), "match_score": ms}, "text": txt})

        w = w_sum / max(1, len(chain))
        final = prev[-1] if prev else ""
        return final, w, {"steps": steps_trace, "records": step_records}


# ---------------------- global API ----------------------
_global_orchestrator = SymphonyOrchestrator(
    verbose=False,
    use_dynamic=True,
    topL=3,
    linucb_alpha=1.0,
    linucb_l2=1.0,
    latency_scale_ms=2000.0,
    latency_penalty=0.2,
    win_bonus=0.5,
    correctness_bonus=0.0,  # 默认不启用 correctness reward（兼容旧 runner）
    incorrect_penalty=0.0,
    plan_k=1,  # 默认不启用 planner（BBH 先稳定评测）
    enable_risk_guard=False,
)


def init(
        *,
        verbose: Optional[bool] = None,
        use_dynamic: Optional[bool] = None,
        topL: Optional[int] = None,
        linucb_alpha: Optional[float] = None,
        linucb_l2: Optional[float] = None,
        plan_k: Optional[int] = None,
        enable_risk_guard: Optional[bool] = None,
        # correctness reward knobs
        correctness_bonus: Optional[float] = None,
        incorrect_penalty: Optional[float] = None,
) -> None:
    """
    Configure global orchestrator WITHOUT clearing registered agents.
    """
    global _global_orchestrator

    if verbose is not None:
        _global_orchestrator.verbose = bool(verbose)

    if use_dynamic is not None:
        _global_orchestrator.use_dynamic = bool(use_dynamic)
        if _global_orchestrator.use_dynamic and _global_orchestrator.selector is None:
            a = float(linucb_alpha) if linucb_alpha is not None else 1.0
            l2 = float(linucb_l2) if linucb_l2 is not None else 1.0
            _global_orchestrator.selector = GlobalLinUCB(d=6, l2=l2, alpha=a)
        if (not _global_orchestrator.use_dynamic) and _global_orchestrator.selector is not None:
            _global_orchestrator.selector = None

    if topL is not None:
        _global_orchestrator.topL = max(1, int(topL))

    if linucb_alpha is not None or linucb_l2 is not None:
        if _global_orchestrator.use_dynamic:
            a = float(linucb_alpha) if linucb_alpha is not None else 1.0
            l2 = float(linucb_l2) if linucb_l2 is not None else 1.0
            _global_orchestrator.selector = GlobalLinUCB(d=6, l2=l2, alpha=a)

    if plan_k is not None:
        _global_orchestrator.plan_k = max(1, int(plan_k))

    if enable_risk_guard is not None:
        _global_orchestrator.enable_risk_guard = bool(enable_risk_guard)
        if _global_orchestrator.enable_risk_guard and RiskAwareGuard is not None and RiskGuardConfig is not None:
            _global_orchestrator.risk_guard = RiskAwareGuard(RiskGuardConfig())
        else:
            _global_orchestrator.risk_guard = None

    if correctness_bonus is not None:
        _global_orchestrator.correctness_bonus = float(correctness_bonus)
    if incorrect_penalty is not None:
        _global_orchestrator.incorrect_penalty = float(incorrect_penalty)


def execute_task(
        task: Task,
        cot_count: int = 3,
        verbose: Optional[bool] = None,
        return_mode: str = "aggregate",
) -> Any:
    if verbose is not None:
        _global_orchestrator.verbose = bool(verbose)
    return _global_orchestrator.execute_task(task, cot_count=cot_count, return_mode=return_mode)


def register_agent(agent: Agent) -> None:
    _global_orchestrator.register_agent(agent)


def get_registered_agents() -> List[Agent]:
    return _global_orchestrator.get_registered_agents()





