"""Core local memory management for task caching and neighbor tracking."""

import time
from collections import deque, defaultdict
from typing import Dict, List, Any

__all__ = ['LocalMemory']


class LocalMemory:
    """Local memory manager for caching tasks and tracking neighbors.
    
    This class provides functionality to cache task results, maintain
    neighbor information, and track neighbor performance scores for
    efficient task delegation and collaboration.
    
    Attributes:
        task_cache (deque): Circular buffer storing recent task results.
        neighbor_data (Dict): Node ID to neighbor metadata mapping.
        neighbor_scores (defaultdict): Node ID to performance score history.
    """
    
    def __init__(self, task_limit: int = 100, neighbor_limit: int = 50):
        """Initialize the local memory with specified limits.
        
        Args:
            task_limit (int, optional): Maximum number of tasks to cache.
                Defaults to 100.
            neighbor_limit (int, optional): Maximum number of neighbors to track.
                Currently unused but reserved for future implementation.
                Defaults to 50.
        """
        self.task_cache = deque(maxlen=task_limit)
        self.neighbor_data = {}  # node_id -> {capabilities, last_seen, score}
        self.neighbor_scores = defaultdict(lambda: deque(maxlen=20))
    
    def store_result(self, result: Dict[str, Any]) -> None:
        """Store a task result in the cache with timestamp.
        
        Args:
            result (Dict[str, Any]): Task result data to store.
        """
        self.task_cache.append({
            "timestamp": int(time.time()),
            **result
        })
    
    def cache_task(self, task_id: str, task_data: Dict) -> None:
        """Cache task data with task ID.
        
        Args:
            task_id (str): Unique task identifier.
            task_data (Dict): Task data to cache.
        """
        self.task_cache.append({
            "task_id": task_id,
            **task_data
        })
    
    def get_recent_tasks(self, n: int = 5) -> List[Dict]:
        """Get the most recent cached tasks.
        
        Args:
            n (int, optional): Number of recent tasks to return.
                Defaults to 5.
        
        Returns:
            List[Dict]: List of recent task data dictionaries.
        """
        return list(self.task_cache)[-n:]
    
    def update_neighbor(self, node_id: str, capabilities: List[str],
                       success: bool) -> None:
        """Update neighbor information and performance score.
        
        Args:
            node_id (str): Unique node identifier.
            capabilities (List[str]): List of neighbor's capabilities.
            success (bool): Whether the last interaction was successful.
        """
        now = int(time.time())
        if node_id not in self.neighbor_data:
            self.neighbor_data[node_id] = {
                "capabilities": capabilities,
                "last_seen": now,
                "score": 0.5
            }
        else:
            self.neighbor_data[node_id]["last_seen"] = now
            self.neighbor_data[node_id]["capabilities"] = capabilities
        
        # Update performance score history
        self.neighbor_scores[node_id].append(1.0 if success else 0.0)
    
    def get_neighbors(self) -> List[str]:
        """Get list of all tracked neighbor IDs.
        
        Returns:
            List[str]: List of neighbor node IDs.
        """
        return list(self.neighbor_data.keys())
    
    def get_neighbor_score(self, node_id: str) -> float:
        """Calculate average performance score for a neighbor.
        
        Args:
            node_id (str): Node ID to calculate score for.
        
        Returns:
            float: Average performance score between 0.0 and 1.0.
                Returns 0.5 if no scores available.
        """
        scores = self.neighbor_scores[node_id]
        if not scores:
            return 0.5
        return round(sum(scores) / len(scores), 3)
    
    def dump(self) -> None:
        """Print memory contents for debugging.
        
        Outputs cached tasks and neighbor information with scores.
        """
        print("\n[LocalMemory Dump]")
        for task in self.task_cache:
            print("- Task:", task)
        for node_id, meta in self.neighbor_data.items():
            score = self.get_neighbor_score(node_id)
            print(f"- Neighbor {node_id[:6]} | Score: {score}")
