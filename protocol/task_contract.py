# protocol/task_contract.py
"""
Task contract and result definitions for distributed task execution.

✅ Symphony 2.0 changes (IMPORTANT):
1) Task legacy-mode now preserves `task_id` from incoming dict instead of generating a new one.
   - Otherwise task_id will change across hops/from_dict calls -> breaks dynamic bandit mapping.
2) Task legacy-mode now preserves `context` if present (so we can carry dispatch_id/task_key).
3) TaskResult now includes `task_id` (dispatch_id / task_key) so requester can update LinUCB.
"""

import uuid
from typing import Dict, Any, List, Optional

__all__ = ["Task", "TaskResult"]


class TaskResult:
    """
    Task execution result container.

    ✅ Symphony 2.0:
    - add task_id: used as "dispatch_id / task_key" for requester-side bandit update
    """

    def __init__(
        self,
        target_id: str,
        executer_id: str,
        result: Any,
        previous_results: Any,
        task_id: str = "",  # ✅ NEW (optional, backward-compatible)
    ) -> None:
        self.target_id = target_id
        self.executer_id = executer_id
        self.result = result
        self.previous_results = previous_results
        self.task_id = task_id or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "executer_id": self.executer_id,
            "result": self.result,
            "previous_results": self.previous_results,
            "task_id": self.task_id,  # ✅ NEW
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskResult":
        return TaskResult(
            target_id=data.get("target_id", ""),
            executer_id=data.get("executer_id", ""),
            result=data.get("result", None),
            previous_results=data.get("previous_results", None),
            task_id=data.get("task_id", ""),  # ✅ NEW
        )

    def __repr__(self) -> str:
        return f"<Result from {self.executer_id} to {self.target_id} task_id={self.task_id}>"


class Task:
    """
    Task contract for distributed computation.

    Supports:
    - High-level API (description/requirements/context/task_id)
    - Legacy API (subtask_id/steps/previous_results/original_problem/final_result/user_id)

    ✅ Symphony 2.0 fixes:
    - legacy mode must keep task_id stable (use provided task_id if present)
    - legacy mode must keep context (for dispatch_id/task_key carrying)
    """

    def __init__(
        self,
        # New API
        description: str = None,
        requirements: List[str] = None,
        context: Dict[str, Any] = None,
        task_id: str = None,
        # Legacy API
        subtask_id: int = None,
        steps: Dict[str, Any] = None,
        previous_results: List[Any] = None,
        original_problem: str = None,
        final_result: str = None,
        user_id: str = None,
    ) -> None:

        # Decide API mode
        if description is not None or requirements is not None:
            # ---------------- High-level ----------------
            self.task_id = task_id or str(uuid.uuid4())
            self.description = description or ""
            self.requirements = requirements or []
            self.context = context or {}

            # legacy-compatible mirrors
            self.subtask_id = subtask_id or 0
            self.steps = steps or {}
            self.previous_results = previous_results or []
            self.original_problem = original_problem or self.description
            self.final_result = final_result or ""
            self.user_id = user_id or "symphony_user"
            self._api_mode = "high_level"
        else:
            # ---------------- Legacy ----------------
            self.subtask_id = subtask_id or 0
            self.steps = steps or {}
            self.previous_results = previous_results or []
            self.original_problem = original_problem or ""
            self.final_result = final_result or ""
            self.user_id = user_id or "symphony_user"

            # ✅ Symphony 2.0: KEEP task_id if provided (do not regenerate!)
            self.task_id = task_id or str(uuid.uuid4())

            # ✅ preserve context if provided, otherwise default legacy marker
            self.context = context or {"mode": "legacy"}

            # best-effort new fields from legacy data
            self.description = self.original_problem or ""
            self.requirements = list(self.steps.keys()) if isinstance(self.steps, dict) else []
            self._api_mode = "legacy"

    def to_dict(self) -> Dict[str, Any]:
        base_dict = {
            "subtask_id": self.subtask_id,
            "steps": self.steps,
            "previous_results": self.previous_results,
            "original_problem": self.original_problem,
            "final_result": self.final_result,
            "user_id": self.user_id,
            # ✅ always include stable ids + context for Symphony 2.0 routing
            "task_id": self.task_id,
            "context": self.context,
            "api_mode": getattr(self, "_api_mode", "legacy"),
            "description": getattr(self, "description", ""),
            "requirements": getattr(self, "requirements", []),
        }
        return base_dict

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Task":
        # Treat as high-level if explicitly present
        if "description" in data or "requirements" in data:
            return Task(
                description=data.get("description"),
                requirements=data.get("requirements", []),
                context=data.get("context", {}),
                task_id=data.get("task_id"),
                subtask_id=data.get("subtask_id"),
                steps=data.get("steps"),
                previous_results=data.get("previous_results"),
                original_problem=data.get("original_problem"),
                final_result=data.get("final_result"),
                user_id=data.get("user_id"),
            )

        # ✅ Legacy mode, but keep task_id + context
        return Task(
            subtask_id=data.get("subtask_id", 0),
            steps=data.get("steps", {}),
            previous_results=data.get("previous_results", []),
            original_problem=data.get("original_problem", ""),
            final_result=data.get("final_result", ""),
            user_id=data.get("user_id", "symphony_user"),
            task_id=data.get("task_id"),              # ✅ KEEP
            context=data.get("context", {"mode": "legacy"}),  # ✅ KEEP
        )

    def __repr__(self) -> str:
        return f"<Task id={self.task_id} subtask_id={self.subtask_id} mode={getattr(self,'_api_mode','legacy')}>"

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)



