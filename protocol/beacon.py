# protocol/beacon.py
import uuid
import time
from typing import Dict, Optional, Any, List


class Beacon:
    """Beacon message for discovering and requesting services in the network."""

    def __init__(
        self,
        sender: str,
        requirement: str,
        task_id: Optional[str] = None,
        ttl: int = 2,

        # ---- Symphony 2.0 additions (all optional for backward compatibility) ----
        task_type: Optional[str] = None,                 # e.g., "math", "code", "medical"
        requirement_embedding: Optional[List[float]] = None,  # task embedding (fixed dim)
        deadline_ms: Optional[int] = None,               # QoS constraint
        priority: Optional[int] = None,                  # e.g., 0~10
        risk_level: Optional[str] = None,                # "low"|"medium"|"high"
        max_candidates: Optional[int] = None,            # stop after K replies (collector side)
        metadata: Optional[Dict[str, Any]] = None,       # extensible extra info
    ) -> None:
        self.beacon_id = str(uuid.uuid4())
        self.sender = sender
        self.task_id = task_id or self.beacon_id
        self.requirement = requirement
        self.ttl = ttl
        self.timestamp = int(time.time())

        # new fields
        self.task_type = task_type
        self.requirement_embedding = requirement_embedding
        self.deadline_ms = deadline_ms
        self.priority = priority
        self.risk_level = risk_level
        self.max_candidates = max_candidates
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "beacon_id": self.beacon_id,
            "sender": self.sender,
            "task_id": self.task_id,
            "requirement": self.requirement,
            "ttl": self.ttl,
            "timestamp": self.timestamp,
        }

        # add optional fields only when present (keeps payload small + backward compatible)
        if self.task_type is not None:
            data["task_type"] = self.task_type
        if self.requirement_embedding is not None:
            data["requirement_embedding"] = self.requirement_embedding
        if self.deadline_ms is not None:
            data["deadline_ms"] = self.deadline_ms
        if self.priority is not None:
            data["priority"] = self.priority
        if self.risk_level is not None:
            data["risk_level"] = self.risk_level
        if self.max_candidates is not None:
            data["max_candidates"] = self.max_candidates
        if self.metadata:
            data["metadata"] = self.metadata

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Beacon":
        beacon = Beacon(
            sender=data.get("sender", "unknown"),
            requirement=data.get("requirement", ""),
            task_id=data.get("task_id"),
            ttl=data.get("ttl", 2),

            # new optional fields
            task_type=data.get("task_type"),
            requirement_embedding=data.get("requirement_embedding"),
            deadline_ms=data.get("deadline_ms"),
            priority=data.get("priority"),
            risk_level=data.get("risk_level"),
            max_candidates=data.get("max_candidates"),
            metadata=data.get("metadata"),
        )
        beacon.beacon_id = data.get("beacon_id", str(uuid.uuid4()))
        beacon.timestamp = data.get("timestamp", int(time.time()))
        return beacon

    def __repr__(self) -> str:
        tt = f", type={self.task_type}" if self.task_type else ""
        dl = f", deadline_ms={self.deadline_ms}" if self.deadline_ms else ""
        return (
            f"<Beacon {self.task_id[:8]} from {self.sender} "
            f"need '{self.requirement}' TTL={self.ttl}{tt}{dl}>"
        )

