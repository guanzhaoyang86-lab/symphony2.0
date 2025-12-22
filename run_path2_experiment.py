# experiments/run_path2_experiment.py
import threading
import time

from agents.agent import Agent
from protocol.task_contract import Task


def loop(agent: Agent, stop_evt: threading.Event):
    while not stop_evt.is_set():
        agent.process_incoming_once(timeout=0.01)


def main(rounds: int = 200):
    host = "127.0.0.1"
    A_port, B_port, C_port = 9311, 9312, 9313

    # --- configs (fully-connected so everyone can send to everyone) ---
    cfgA = {
        "node_id": "A",
        "network": {"host": host, "port": A_port},
        "neighbours": [("B", host, B_port), ("C", host, C_port)],
        "capabilities": ["general-reasoning"],
        "response_timeout": 1,
        "linucb_alpha": 1.0,
    }
    cfgB = {
        "node_id": "B",
        "network": {"host": host, "port": B_port},
        "neighbours": [("A", host, A_port), ("C", host, C_port)],
        "capabilities": ["general-reasoning"],
        "sim_delay_ms": 30,        # ✅ B 快
        "sim_fail_rate": 0.0,
        "response_timeout": 1,
    }
    cfgC = {
        "node_id": "C",
        "network": {"host": host, "port": C_port},
        "neighbours": [("A", host, A_port), ("B", host, B_port)],
        "capabilities": ["general-reasoning"],
        "sim_delay_ms": 250,       # ✅ C 慢
        "sim_fail_rate": 0.0,
        "response_timeout": 1,
    }

    # --- start agents ---
    A = Agent(cfgA)
    B = Agent(cfgB)
    C = Agent(cfgC)

    stop = threading.Event()
    tb = threading.Thread(target=loop, args=(B, stop), daemon=True)
    tc = threading.Thread(target=loop, args=(C, stop), daemon=True)
    tb.start()
    tc.start()

    # requester(A) 不开后台线程也行：我们在等待时主动 poll process_incoming_once
    selection = {"B": 0, "C": 0}

    for t in range(rounds):
        # ✅ 可选：制造非平稳环境（第 100 轮交换快慢）
        if t == rounds // 2:
            B._sim_delay_ms, C._sim_delay_ms = C._sim_delay_ms, B._sim_delay_ms
            print(f"[DRIFT] swapped delays: B={B._sim_delay_ms}ms, C={C._sim_delay_ms}ms")

        task = Task(
            task_id=f"task_{t}",
            subtask_id=1,
            steps={"1": ["do something", "general-reasoning"]},
            previous_results=[],
            original_problem="demo",
            final_result="",
            user_id="exp",
            context={"domain": "demo"},
        )

        dispatch_id, chosen = A.assign_task(task, topL=2)
        if chosen in selection:
            selection[chosen] += 1

        # wait until this dispatch is updated (pending removed)
        t0 = time.time()
        while dispatch_id in A._pending and (time.time() - t0) < 5:
            A.process_incoming_once(timeout=0.02)

    stop.set()
    time.sleep(0.2)

    print("selection counts:", selection)
    print("theta_hat:", A._selector.theta_hat())


if __name__ == "__main__":
    main(rounds=200)
