# agents/agent.py
"""
Agent implementation for distributed computation in Symphony Network.

✅ Symphony 2.0 Key Additions:
- dynamic_state reporting in BeaconResponse
- requester-side Dynamic Beacon Selection via LinUCB (GlobalLinUCB)
- strict task_key/dispatch_id propagation to guarantee bandit update hits
"""

from typing import Dict, List, Tuple, Any
import time
import os
import json
from typing import Any, Dict, Optional, List
from models.base_loader import BaseModel

try:
    import requests
except ImportError:
    requests = None
from models.base_loader import BaseModel  # 失败就直接报错，别 fallback

# -------------------- LinUCB selector --------------------
from core.linucb_selector import GlobalLinUCB, build_x

try:
    from protocol.beacon import Beacon
    from protocol.response import BeaconResponse
except ImportError:
    Beacon = None
    BeaconResponse = None

try:
    from core.capability import CapabilityManager
    from core.memory import LocalMemory
except ImportError:
    class CapabilityManager:
        def __init__(self, capabilities):
            self.capabilities = capabilities

        def match(self, requirement):
            return 0.5 if requirement in " ".join(self.capabilities) else 0.1


    class LocalMemory:
        def __init__(self):
            pass

try:
    from protocol.task_contract import TaskResult, Task
except ImportError:
    TaskResult = None
    Task = None

try:
    from infra.ISEP import ISEPClient
    from infra.network_adapter import NetworkAdapter
except ImportError:
    ISEPClient = None
    NetworkAdapter = None


def _is_openai_spec(spec: str) -> bool:
    return isinstance(spec, str) and spec.startswith("openai:")


def _parse_openai_spec(spec: str):
    rest = spec[len("openai:"):]
    if "#model=" in rest:
        api_base, model = rest.split("#model=", 1)
    else:
        api_base, model = rest, None
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    return api_base, model


class OpenAICompatModel:
    """Minimal OpenAI-compatible HTTP client (optional)."""

    def __init__(self, api_base: str, model: str, system_prompt: str = ""):
        if requests is None:
            raise RuntimeError("requests not installed. `pip install requests`")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or ""
        self._url_chat = f"{self.api_base}/chat/completions"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}",
        }

    def generate(self, text: str, max_tokens: int = 10124, temperature: float = 0.2) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        r = requests.post(self._url_chat, headers=self._headers, data=json.dumps(payload), timeout=180)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

    def extract_task(self, user_input: str):
        return "", user_input.strip(), True

    def generate_task_dag(self, background: str, question: str, user_input: str, domain: str):
        steps = {"1": ("Solve the problem carefully", "general-reasoning")}
        return steps, True


class Agent:
    def __init__(
            self,
            node_id: str = None,
            capabilities: List[str] = None,
            system_prompt: str = None,
            base_model: str = None,
            gpu_id: int = 0,
            config: Dict[str, Any] = None,
    ) -> None:
        if config is not None:
            self._init_from_config(config)
        else:
            self._init_from_params(node_id, capabilities, system_prompt, base_model, gpu_id)

        # ✅ Symphony2.0: global LinUCB + pending buffer (requester side)
        self._selector = GlobalLinUCB(d=6, l2=1.0, alpha=float(getattr(self, "_linucb_alpha", 1.0)))
        self._pending: Dict[str, Dict[str, Any]] = {}  # dispatch_id -> {x,t0,executor}

    # ---------------- dynamic state ----------------
    def _init_dynamic_state(self, max_inflight: int = 1, latency_ema_init_ms: float = 500.0,
                            latency_ema_beta: float = 0.2):
        self._inflight = 0
        self._max_inflight = max(1, int(max_inflight))
        self._latency_ema_ms = float(latency_ema_init_ms)
        self._latency_beta = float(latency_ema_beta)
        self._reputation = 0.5

    def _init_from_config(self, config: Dict[str, Any]) -> None:
        self.agent_id = config["node_id"]
        self.gpu_id = config.get("gpu_id", 0)
        self.system_prompt = config.get("sys_prompt", "You are a helpful AI assistant.")
        self.capabilities = config.get("capabilities", [])
        self._linucb_alpha = float(config.get("linucb_alpha", 1.0))

        bm = config.get("base_model")
        if bm == "test":
            self.base_model = None
        elif isinstance(bm, str) and _is_openai_spec(bm):
            api_base, model = _parse_openai_spec(bm)
            self.base_model = OpenAICompatModel(api_base, model, self.system_prompt)
        elif BaseModel is not None and bm:
            self.base_model = BaseModel(bm, self.system_prompt, device=f"cuda:{self.gpu_id}")
        else:
            self.base_model = None

        self.memory = LocalMemory()
        self.capability_manager = CapabilityManager(self.capabilities)

        self._init_dynamic_state(
            max_inflight=config.get("max_inflight", 1),
            latency_ema_init_ms=config.get("latency_ema_init_ms", 500.0),
            latency_ema_beta=config.get("latency_ema_beta", 0.2),
        )

        if "neighbours" in config and NetworkAdapter is not None and ISEPClient is not None:
            self.network = NetworkAdapter(self.agent_id, config)
            self.isep_client = ISEPClient(self.agent_id, self.network,
                                          response_timeout=config.get("response_timeout", 1))
            self._initialize_neighbors(config["neighbours"])
        else:
            self.network = None
            self.isep_client = None

    def _init_from_params(
            self,
            node_id: str,
            capabilities: List[str],
            system_prompt: str,
            base_model: Any,
            gpu_id: int,
            config: Optional[Dict[str, Any]] = None,  # ✅ 新增：可选 config，不改调用也能跑
    ) -> None:
        if node_id is None:
            raise ValueError("node_id is required")

        # ✅ 1) 先保证 config 一定存在且是 dict
        config = config or {}

        # ✅ 2)（可选但推荐）如果 base_model 对象自带 config，把它合并进来
        #    注意：不覆盖你显式传入的 config key（显式优先）
        try:
            bm_cfg = getattr(base_model, "config", None)
            if bm_cfg is not None:
                if not isinstance(bm_cfg, dict):
                    try:
                        bm_cfg = vars(bm_cfg)
                    except Exception:
                        bm_cfg = {}
                # base_model 的配置作为“默认值”，config 里已有的优先
                for k, v in (bm_cfg or {}).items():
                    config.setdefault(k, v)
        except Exception:
            pass

        self.agent_id = node_id
        self.gpu_id = gpu_id
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.capabilities = capabilities or []
        self._linucb_alpha = float(config.get("linucb_alpha", 1.0))

        # ✅ 3) 这里就别再写 if config else 了，直接 get 默认值
        self.max_tokens = int(config.get("max_tokens", 512))
        self.temperature = float(config.get("temperature", 0.2))
        self.top_p = float(config.get("top_p", 0.9))

        if base_model and base_model != "test":
            if _is_openai_spec(base_model):
                api_base, model = _parse_openai_spec(base_model)
                self.base_model = OpenAICompatModel(api_base, model, self.system_prompt)
            elif BaseModel is not None:
                self.base_model = BaseModel(base_model, self.system_prompt, device=f"cuda:{self.gpu_id}")
            else:
                self.base_model = None
        else:
            self.base_model = None

        self.memory = LocalMemory()
        self.capability_manager = CapabilityManager(self.capabilities)
        self._init_dynamic_state()

        self.network = None
        self.isep_client = None

        # auto register (local orchestrator path)
        try:
            import symphony
            symphony.register_agent(self)
        except Exception:
            pass

    def _initialize_neighbors(self, neighbors: List[Tuple[str, str, str]]) -> None:
        for nid, host, port in neighbors:
            self.network.add_neighbor(nid, host, int(port))

    # ---------------- execution with instrumentation ----------------
    def _execute_subtask(self, task: Task) -> Task:
        sid = str(task["subtask_id"])
        instruction = task["steps"][sid][0]
        previous_context = " ".join(task["previous_results"])

        # ✅ BBH prompt 透传：不要再包一层 system_prompt/context/boxed
        if isinstance(instruction, str) and (
                "Final answer:" in instruction
                or "Task input:" in instruction
                or "Big-Bench Hard" in instruction
        ):
            prompt = instruction.strip()
        else:
            prompt = self._build_task_description(instruction, previous_context)

        if self.base_model is None:
            # 建议：不要静默 simulated，直接报错更安全（除非你明确要模拟）
            raise RuntimeError(f"Agent {self.agent_id} base_model is None (SIMULATED). Check model loading.")
        else:
            try:
                result = self.base_model.generate(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            except TypeError:
                # 兼容不支持 kwargs 的 BaseModel
                result = self.base_model.generate(prompt)

        task["previous_results"].append(f"{instruction} Answer: {result}")

        # 注意：你这里原来用 task["subtask_id"] == len(task["steps"])，如果 subtask_id 是 str 会出坑
        try:
            is_last = int(task["subtask_id"]) >= len(task["steps"])
        except Exception:
            is_last = True

        if is_last:
            task["final_result"] = result

        task["subtask_id"] = int(task.get("subtask_id", 0)) + 1
        return Task.from_dict(task)

    def execute_task(self, task):
        """
        Legacy API shim for Symphony orchestrator.

        symphony._execute_subtask_on_agent(...) 会构造一个 dict 形式的 agent_task，
        然后调用 agent.execute_task(agent_task)。

        这里我们按 steps 顺序执行，最终返回 final_result 字符串。
        """
        # 允许 Task / dict 两种
        try:
            t = task.to_dict() if hasattr(task, "to_dict") else dict(task)
        except Exception:
            t = task

        steps = t.get("steps", {}) or {}
        if not isinstance(steps, dict) or len(steps) == 0:
            return ""

        # subtask_id 统一成 int
        try:
            t["subtask_id"] = int(t.get("subtask_id", 1))
        except Exception:
            t["subtask_id"] = 1

        # 逐步执行
        # 你的 _execute_subtask 里会 task["subtask_id"] += 1，并在最后写 final_result
        while t.get("subtask_id", 1) <= len(steps):
            out = self._execute_subtask(t)
            # _execute_subtask 可能返回 Task-like
            try:
                t = out.to_dict() if hasattr(out, "to_dict") else dict(out)
            except Exception:
                t = out

        final_result = t.get("final_result", "")
        if isinstance(final_result, str):
            return final_result.strip()
        return final_result

    def get_dynamic_state(self) -> dict:
        inflight = int(getattr(self, "_inflight", 0))
        max_inflight = int(getattr(self, "_max_inflight", 1))
        latency_ms = float(getattr(self, "_latency_ema_ms", 500.0))
        reputation = float(getattr(self, "_reputation", 0.5))
        load = max(0.0, min(1.0, float(inflight) / float(max(1, max_inflight))))
        return {
            "available": inflight < max_inflight,
            "load": load,
            "latency_ms": latency_ms,
            "reputation": max(0.0, min(1.0, reputation)),
        }

    def _decompose_task(self, task: Task) -> Task:
        user_input = task["original_problem"]
        if self.base_model is None:
            task["steps"] = {"1": ("Solve the problem carefully", "general-reasoning")}
            task["subtask_id"] += 1
            return Task.from_dict(task)

        bg, q, ok = self.base_model.extract_task(user_input)
        if ok:
            task["previous_results"].append(bg)
            steps, dag_ok = self.base_model.generate_task_dag(bg, q, user_input, "math")
            if dag_ok:
                task["steps"] = steps
        task["subtask_id"] += 1
        return Task.from_dict(task)

    def _build_task_description(self, instruction: str, context: str) -> str:
        return (
            f"{self.system_prompt}\n"
            f"Context: {context}\n"
            f"Task: {instruction}\n"
            f"You may think step-by-step, but output ONLY the final answer.\n"
            f"The LAST line must start with exactly: Final answer:\n"
            f"Final answer:"
        )

    # ---------------- beacon handling (executor side) ----------------
    def handle_beacon(self, sender_id: str, beacon: Any) -> None:
        match_score = self.capability_manager.match(beacon["requirement"])

        load = max(0.0, min(1.0, float(self._inflight) / float(max(1, self._max_inflight))))
        available = self._inflight < self._max_inflight

        dynamic_state = {
            "load": load,
            "queue_len": int(self._inflight),
            "latency_ms": float(self._latency_ema_ms),
            "reputation": float(self._reputation),
        }
        estimate_cost = float(self._latency_ema_ms) / 1000.0

        response = BeaconResponse(
            responder_id=self.agent_id,
            task_id=beacon["task_id"],
            match_score=match_score,
            estimate_cost=estimate_cost,
            available=available,
            dynamic_state=dynamic_state,
        )

        payload = response.to_dict() if hasattr(response, "to_dict") else response
        self.isep_client.send_response(sender_id, "beacon_response", payload)

    # ---------------- Dynamic Beacon Selection (requester side) ----------------
    def assign_task(self, task: Task, topL: int = 5) -> None:
        if self.isep_client is None:
            raise RuntimeError("isep_client not initialized; cannot assign_task in local mode.")

        tid = getattr(task, "task_id", None) or task.get("task_id", "task")
        sub_id = int(getattr(task, "subtask_id", task.get("subtask_id", 0)))
        task_key = f"{tid}:{sub_id}"  # ✅ dispatch_id / task_key (MUST stay consistent)

        # ✅ write dispatch_id into task.context so executor can return it
        ctx = getattr(task, "context", None) or task.get("context", {}) or {}
        ctx["dispatch_id"] = task_key
        try:
            task.context = ctx
        except Exception:
            task["context"] = ctx

        requirement = getattr(task, "steps", task.get("steps"))[str(sub_id)][1]

        beacon = Beacon(sender=self.agent_id, task_id=task_key, requirement=requirement, ttl=2)

        responses = self.isep_client.broadcast_and_collect(beacon)  # ✅ full dict list

        # candidate filter: availability + threshold
        cand: List[Dict[str, Any]] = []
        for r in responses:
            ms = float(r.get("match_score", 0.0))
            av = bool(r.get("available", True))
            if ms < 0.3:
                continue
            if not av:
                continue
            cand.append(r)

        if not cand:
            raise RuntimeError(f"No available candidates for requirement={requirement}")

        # Stage-A: Top-L by match_score
        cand.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
        cand = cand[: max(1, int(topL))]

        # Stage-B: LinUCB
        candidates_x: List[Tuple[str, List[float]]] = []
        x_map: Dict[str, List[float]] = {}

        for r in cand:
            rid = str(r.get("responder_id", ""))
            x = build_x(
                match_score=float(r.get("match_score", 0.0)),
                dynamic_state=r.get("dynamic_state", {}) or {},
                available=bool(r.get("available", True)),
                latency_scale_ms=2000.0,
            )
            candidates_x.append((rid, x))
            x_map[rid] = x

        chosen_id = self._selector.select(candidates_x)
        x_chosen = x_map[chosen_id]

        # Save pending for online update when result arrives
        self._pending[task_key] = {"x": x_chosen, "t0": time.time(), "executor": chosen_id}

        # delegate
        self.isep_client.delegate_task(chosen_id, task)

    # ---------------- bandit update (requester side) ----------------
    def on_task_result(self, result: Dict[str, Any]) -> None:
        task_key = str(result.get("task_id", ""))
        if not task_key or task_key not in self._pending:
            return

        rec = self._pending.pop(task_key)
        x = rec["x"]
        t0 = float(rec["t0"])
        latency_ms = (time.time() - t0) * 1000.0

        text = str(result.get("result", ""))
        ok = (text.strip() != "") and (not text.startswith("[ERROR]")) and (not text.startswith("[AGENT_ERROR]"))

        # ✅ reward bounded in [0,1] (align with common LinUCB assumptions)
        lat_norm = min(1.0, latency_ms / 2000.0)
        reward = (1.0 if ok else 0.0) - 0.2 * (lat_norm ** 0.5)
        reward = max(0.0, min(1.0, reward))  # ✅ CLIP

        self._selector.update(x, reward)

        # optional reputation proxy
        self._reputation = max(0.0, min(1.0, 0.95 * self._reputation + 0.05 * (1.0 if ok else 0.0)))

    # ---------------- runtime helper loop ----------------
    def process_incoming_once(self, timeout: float = 0.05) -> None:
        """
        Call this in your runtime loop (both requester & executor) to:
        - handle incoming beacons (executor -> respond)
        - handle incoming tasks (executor -> execute -> submit_result)
        - handle incoming results (requester -> update bandit)
        """
        if self.isep_client is None:
            return

        # 1) beacons
        sid, _, beacon = self.isep_client.receive_beacon(timeout=timeout)
        if beacon is not None:
            self.handle_beacon(sid, beacon)

        # 2) tasks
        sid, _, task = self.isep_client.receive_task(timeout=timeout)
        if task is not None:
            task_obj = Task.from_dict(task) if isinstance(task, dict) else task

            # ✅ read dispatch_id from context; never "guess"
            ctx = task_obj.get("context", {}) or {}
            dispatch_id = str(ctx.get("dispatch_id", task_obj.get("task_id", "")))

            done_text = self.execute_task(task_obj)  # str
            self.isep_client.submit_result(
                target_id=sid,
                result=done_text,
                previous_results=[],
                task_id=dispatch_id,
            )

        # 3) results
        sid, _, res = self.isep_client.receive_result(timeout=timeout)
        if res is not None:
            self.on_task_result(res)







