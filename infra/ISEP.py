# infra/ISEP.py
"""
Inter-node Service Exchange Protocol (ISEP) Client

✅ Symphony 2.0 changes:
- broadcast_and_collect returns FULL response dicts (incl. dynamic_state)
- normalize inbound BeaconResponse/TaskResult into dict
- submit_result carries task_id (dispatch_id/task_key) for bandit update
- pop pending_tasks after collection to avoid memory leak
"""

from typing import Dict, List, Any
import time
from queue import Queue

from protocol.beacon import Beacon
from protocol.response import BeaconResponse
from protocol.task_contract import TaskResult, Task
from infra.network_adapter import NetworkAdapter


class ISEPClient:
    def __init__(self, node_id: str, network_adapter: NetworkAdapter, response_timeout: int = 1):
        self.node_id = node_id
        self.network = network_adapter
        self.response_timeout = int(response_timeout)

        # task_id -> list[response_dict]
        self.pending_tasks: Dict[str, List[Dict[str, Any]]] = {}

        self.beacon_queue = Queue()
        self.subtask_queue = Queue()
        self.task_result_queue = Queue()

        # handlers
        self.network.register_handler("beacon", self._handle_beacon)
        self.network.register_handler("beacon_response", self._handle_beacon_response)
        self.network.register_handler("task", self._handle_task)
        self.network.register_handler("task_result", self._handle_task_result)

    # ---------------------------------------------------------------------
    # ✅ Symphony2.0: return full responses (not just (id, score))
    # ---------------------------------------------------------------------
    def broadcast_and_collect(self, beacon: Beacon) -> List[Dict[str, Any]]:
        tid = str(beacon.task_id)
        self.pending_tasks[tid] = []

        self.network.broadcast("beacon", beacon)
        time.sleep(self.response_timeout)

        replies = list(self.pending_tasks.get(tid, []))

        # ✅ important: prevent memory leak
        try:
            self.pending_tasks.pop(tid, None)
        except Exception:
            pass

        return replies

    def send_response(self, target_id: str, msg_type: str, response: Any) -> None:
        self.network.send(target_id, msg_type, response)

    def delegate_task(self, executor_id: str, task: Task) -> str:
        self.network.send(executor_id, "task", task)
        return "ok"

    # ---------------------------------------------------------------------
    # ✅ Symphony2.0: include task_id (dispatch_id/task_key) in TaskResult
    # ---------------------------------------------------------------------
    def submit_result(
        self,
        target_id: str,
        result: Any,
        previous_results: Any,
        task_id: str = "",
    ) -> None:
        task_result = TaskResult(
            target_id=target_id,
            executer_id=self.node_id,
            result=result,
            previous_results=previous_results,
            task_id=task_id,  # ✅ NEW
        )
        self.network.send(target_id, "task_result", task_result)

    # -------------------------- handlers --------------------------

    def _handle_beacon(self, sender_id: str, beacon: Any) -> None:
        self.beacon_queue.put((sender_id, "beacon", beacon))

    def _handle_beacon_response(self, sender_id: str, response: Any) -> None:
        # normalize to dict
        if isinstance(response, BeaconResponse):
            resp = response.to_dict()
        elif isinstance(response, dict):
            resp = response
        else:
            try:
                resp = response.to_dict()
            except Exception:
                return

        tid = str(resp.get("task_id", ""))
        if tid and tid in self.pending_tasks:
            self.pending_tasks[tid].append(resp)

    def _handle_task(self, sender_id: str, task: Any) -> None:
        self.subtask_queue.put((sender_id, "task", task))

    def _handle_task_result(self, sender_id: str, result: Any) -> None:
        # normalize to dict
        if isinstance(result, TaskResult):
            res = result.to_dict()
        elif isinstance(result, dict):
            res = result
        else:
            try:
                res = result.to_dict()
            except Exception:
                res = {"executer_id": sender_id, "result": str(result), "task_id": ""}

        self.task_result_queue.put((sender_id, "task_result", res))

    # -------------------------- receive APIs --------------------------

    def receive_beacon(self, timeout=None):
        try:
            return self.beacon_queue.get(timeout=timeout)
        except Exception:
            return None, None, None

    def receive_task(self, timeout=None):
        try:
            return self.subtask_queue.get(timeout=timeout)
        except Exception:
            return None, None, None

    def receive_result(self, timeout=None):
        try:
            return self.task_result_queue.get(timeout=timeout)
        except Exception:
            return None, None, None



