"""
FastAPI backend that wraps Traffic_Four_Way_Intersection_Environment and
serves step-by-step snapshots to the React intersection diagram.

Run after copying this file into traffic_rl_ui/ (next to environment.py):

    pip install fastapi "uvicorn[standard]"
    uvicorn api:app --reload --port 8000

Then point the frontend's API_BASE at http://127.0.0.1:8000
"""

import os
import pickle
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import Traffic_Four_Way_Intersection_Environment
from discretize import discretize_state
from agents import DQN

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

MAX_QUEUE = 20
NUM_BUCKETS = 2
NUM_ACTIONS = 12
MODEL_DIR = "saved_models"

MOVEMENT_TO_LANE = {
    "N_W": "N1", "N_S": "N2", "N_E": "N3",
    "S_W": "S1", "S_N": "S2", "S_E": "S3",
    "E_N": "E1", "E_W": "E2", "E_S": "E3",
    "W_N": "W1", "W_E": "W2", "W_S": "W3",
}

MOVEMENTS = [
    "N_W", "N_S", "N_E",
    "S_W", "S_N", "S_E",
    "E_N", "E_S", "E_W",
    "W_N", "W_S", "W_E",
]
INDEX_TO_LANE = [MOVEMENT_TO_LANE[m] for m in MOVEMENTS]

OUTGOING_INDEX_TO_DIR = {12: "N", 13: "S", 14: "E", 15: "W"}

# ─────────────────────────────────────────────────────────────────────────
# Model loading (all optional — Baseline always works with no files needed)
# ─────────────────────────────────────────────────────────────────────────

def _try_load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


Q_TABLE = _try_load(os.path.join(MODEL_DIR, "q_table.pkl"))
SARSA_TABLE = _try_load(os.path.join(MODEL_DIR, "sarsa_table.pkl"))

DQN_MODEL: Optional[DQN] = None
dqn_path = os.path.join(MODEL_DIR, "dqn.pt")
if os.path.exists(dqn_path):
    DQN_MODEL = DQN(state_size=16, action_size=NUM_ACTIONS)
    DQN_MODEL.load_state_dict(torch.load(dqn_path, map_location="cpu"))
    DQN_MODEL.eval()

ALGORITHMS = {
    "Baseline": True,
    "Q-learning": Q_TABLE is not None,
    "SARSA": SARSA_TABLE is not None,
    "DQN": DQN_MODEL is not None,
}

WAITING_COMPARISON_EPISODES = 100


def pick_action(algorithm: str, state: np.ndarray):
    """Returns (action_index, decision_info) for the given controller (no exploration).
    decision_info explains *why* that lane was picked -- used by the details panel.
    Falls back to Baseline if the requested algorithm has no trained weights."""
    if algorithm == "Q-learning" and Q_TABLE is not None:
        key = discretize_state(state, MAX_QUEUE, NUM_BUCKETS)
        values = Q_TABLE.get(key, np.zeros(NUM_ACTIONS))
        action = int(np.argmax(values))
        decision = {
            "type": "q-table",
            "source": "Q-learning table lookup",
            "stateBucket": list(key),
            "values": [round(float(v), 2) for v in values],
            "chosenIndex": action,
            "seenBucketBefore": key in Q_TABLE,
        }
        return action, decision

    if algorithm == "SARSA" and SARSA_TABLE is not None:
        key = discretize_state(state, MAX_QUEUE, NUM_BUCKETS)
        values = SARSA_TABLE.get(key, np.zeros(NUM_ACTIONS))
        action = int(np.argmax(values))
        decision = {
            "type": "q-table",
            "source": "SARSA table lookup",
            "stateBucket": list(key),
            "values": [round(float(v), 2) for v in values],
            "chosenIndex": action,
            "seenBucketBefore": key in SARSA_TABLE,
        }
        return action, decision

    if algorithm == "DQN" and DQN_MODEL is not None:
        with torch.no_grad():
            state_t = torch.tensor(state / MAX_QUEUE, dtype=torch.float32).unsqueeze(0)
            q_values = DQN_MODEL(state_t).squeeze(0)
            action = int(q_values.argmax().item())
        decision = {
            "type": "dqn",
            "source": "DQN forward pass",
            "values": [round(float(v), 2) for v in q_values.tolist()],
            "chosenIndex": action,
        }
        return action, decision

    # Baseline: always serve the largest incoming queue
    action = int(np.argmax(state[:NUM_ACTIONS]))
    decision = {
        "type": "rule",
        "source": "Baseline rule",
        "rule": "Always serve the largest incoming queue",
        "values": [int(v) for v in state[:NUM_ACTIONS]],
        "chosenIndex": action,
    }
    return action, decision


# ─────────────────────────────────────────────────────────────────────────
# Snapshot helpers
# ─────────────────────────────────────────────────────────────────────────

def build_details(info: dict, decision: dict):
    state_before = info["state_before"]
    state_after = info["state_after"]
    arrivals = info["arrivals"]
    outgoing_departures = info["outgoing_departures"]
    selected_index = info["selected_index"]
    outgoing_index = info["outgoing_index"]

    lanes = []
    for i, movement in enumerate(MOVEMENTS):
        lanes.append({
            "lane": INDEX_TO_LANE[i],
            "movement": movement,
            "before": int(state_before[i]),
            "arrived": int(arrivals[i]),
            "passed": int(info["cars_passed"]) if i == selected_index else 0,
            "blocked": int(info["blocked_by_outgoing"]) if i == selected_index else 0,
            "after": int(state_after[i]),
            "wasSelected": i == selected_index,
        })

    outgoing = []
    for j, direction in OUTGOING_INDEX_TO_DIR.items():
        i = j
        outgoing.append({
            "direction": direction,
            "before": int(state_before[i]),
            "in": int(info["cars_passed"]) if i == outgoing_index else 0,
            "out": int(outgoing_departures[i - 12]),
            "after": int(state_after[i]),
        })

    agent_decision = dict(decision)
    if "values" in agent_decision:
        agent_decision["values"] = [
            {"lane": INDEX_TO_LANE[i], "movement": MOVEMENTS[i], "value": v}
            for i, v in enumerate(agent_decision["values"])
        ]

    return {
        "agentDecision": agent_decision,
        "selectedLaneBreakdown": {
            "lane": INDEX_TO_LANE[selected_index],
            "before": int(info["selected_queue_before_green"]),
            "capacity": int(info["cars_passed"]) + int(info["blocked_by_outgoing"]),
            "passed": int(info["cars_passed"]),
            "blocked": int(info["blocked_by_outgoing"]),
            "after": int(state_after[selected_index]),
        },
        "lanes": lanes,
        "outgoing": outgoing,
        "blockedArrivals": int(info["blocked_arrivals"]),
        "carsLeftSystem": int(info["cars_left_system"]),
    }


def make_snapshot(episode, step, algorithm, state, reward, done,
                   selected_lane=None, exit_direction=None, cars_passed=0,
                   total_queue=None, details=None):
    incoming = {INDEX_TO_LANE[i]: int(state[i]) for i in range(12)}
    outgoing = {OUTGOING_INDEX_TO_DIR[i]: int(state[i]) for i in range(12, 16)}
    if total_queue is None:
        total_queue = int(sum(incoming.values()) + sum(outgoing.values()))

    return {
        "episode": episode,
        "step": step,
        "algorithm": algorithm,
        "selectedLane": selected_lane,
        "exitDirection": exit_direction,
        "reward": round(float(reward), 2),
        "carsPassed": int(cars_passed),
        "totalQueue": total_queue,
        "incoming": incoming,
        "outgoing": outgoing,
        "done": bool(done),
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────
# Queue-per-step comparison (Baseline vs Q-learning vs SARSA vs DQN)
#
# Q-learning/SARSA/DQN are compared using their last WAITING_COMPARISON_EPISODES
# *training* episodes, read straight from the saved history pickles
# (saved_models/q_history.pkl etc. -- each has a "queues" list of one
# total-queue value per training episode). Baseline never trains, so it has
# no history file -- it still gets a fresh no-exploration eval rollout.
#
# Caveat: unlike a fresh eval rollout, these training episodes still have
# ~epsilon_end exploration noise (random actions), so this is "how the
# policy looked near the end of training," not a clean argmax-only
# comparison. Every training episode runs for exactly MAX_STEPS steps
# (environment truncates, never terminates early), so dividing each
# episode's total queue by MAX_STEPS gives avg queue per step -- same units
# as before.
# ─────────────────────────────────────────────────────────────────────────

MAX_STEPS = 100  # must match max_steps used in agents.py training calls

HISTORY_FILES = {
    "Q-learning": "q_history.pkl",
    "SARSA": "sarsa_history.pkl",
    "DQN": "dqn_history.pkl",
}


def _run_policy_rollout(algorithm: str, num_episodes: int = WAITING_COMPARISON_EPISODES):
    """Fresh, no-exploration evaluation rollout. Used only for Baseline,
    which has no training history to read from. Returns each episode's
    average total_system_queue per step."""
    episode_avgs = []
    for _ in range(num_episodes):
        env = Traffic_Four_Way_Intersection_Environment(use_argmax_policy=False)
        state, _ = env.reset()

        total_queue = 0
        steps = 0
        done = False

        while not done:
            action, _ = pick_action(algorithm, state)
            state, _, terminated, truncated, info = env.step(action)
            total_queue += info["total_system_queue"]
            steps += 1
            done = terminated or truncated

        episode_avgs.append(total_queue / max(steps, 1))

    return episode_avgs


def _load_training_queue_avgs(algorithm: str, num_episodes: int = WAITING_COMPARISON_EPISODES):
    """Average total_system_queue per step, per episode, taken from the last
    `num_episodes` of TRAINING (not a fresh rollout). Returns [] if no
    history file was found."""
    path = os.path.join(MODEL_DIR, HISTORY_FILES[algorithm])
    history = _try_load(path)
    if history is None:
        return []
    queue_totals = history["queues"][-num_episodes:]
    return [q / MAX_STEPS for q in queue_totals]


def get_waiting_comparison():
    """Average queue per step, per algorithm. Q-learning/SARSA/DQN come from
    their last WAITING_COMPARISON_EPISODES training episodes; Baseline comes
    from a fresh no-exploration eval rollout (it has no training history).
    Field name `avgWaiting` kept as-is so the frontend doesn't need a
    contract change -- but this is a queue-occupancy proxy, not literal
    per-car waiting time."""
    results = {}
    for algorithm in ["Baseline", "Q-learning", "SARSA", "DQN"]:
        if not ALGORITHMS.get(algorithm):
            results[algorithm] = {"available": False, "avgWaiting": None, "episodes": 0}
            continue
        if algorithm == "Baseline":
            scores = _run_policy_rollout(algorithm)
        else:
            scores = _load_training_queue_avgs(algorithm)
        results[algorithm] = {
            "available": True,
            "avgWaiting": round(float(np.mean(scores)), 2),
            "episodes": len(scores),
        }
    return results


class Session:
    """Single in-memory episode + snapshot history (this is a local dev tool,
    not a multi-user service, so one global session is enough)."""

    def __init__(self):
        self.env: Optional[Traffic_Four_Way_Intersection_Environment] = None
        self.algorithm = "Baseline"
        self.episode = 0
        self.history = []
        self.index = -1

    def new_episode(self, algorithm: str):
        self.algorithm = algorithm
        self.env = Traffic_Four_Way_Intersection_Environment(use_argmax_policy=False)
        state, _ = self.env.reset()
        self.episode += 1
        snap = make_snapshot(self.episode, 0, algorithm, state, reward=0, done=False)
        self.history = [snap]
        self.index = 0
        return snap

    def step_forward(self):
        if self.env is None:
            raise HTTPException(400, "No active episode. Call /episode/new first.")

        if self.index < len(self.history) - 1:
            self.index += 1
            return self.history[self.index]

        current_state = self.env.state
        action, decision = pick_action(self.algorithm, current_state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        snap = make_snapshot(
            self.episode,
            self.env.current_step,
            self.algorithm,
            next_state,
            reward=reward,
            done=done,
            selected_lane=MOVEMENT_TO_LANE[info["selected_movement"]],
            exit_direction=info["outgoing_movement"].split("_")[0],
            cars_passed=info["cars_passed"],
            total_queue=info["total_system_queue"],
            details=build_details(info, decision),
        )
        self.history.append(snap)
        self.index = len(self.history) - 1
        return snap

    def step_back(self):
        if self.index <= 0:
            raise HTTPException(400, "Already at the first step of this episode.")
        self.index -= 1
        return self.history[self.index]


session = Session()

# ─────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Traffic RL Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewEpisodeRequest(BaseModel):
    algorithm: str = "Baseline"


@app.get("/algorithms")
def get_algorithms():
    return ALGORITHMS


@app.get("/waiting-comparison")
def waiting_comparison():
    """Average queue per step per algorithm, all from fresh no-exploration
    evaluation rollouts under identical conditions. Queue-occupancy proxy
    for waiting time, not literal per-car wait time."""
    return get_waiting_comparison()


@app.post("/episode/new")
def episode_new(req: NewEpisodeRequest):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(400, f"Unknown algorithm '{req.algorithm}'.")
    return session.new_episode(req.algorithm)


@app.post("/episode/step")
def episode_step():
    return session.step_forward()


@app.post("/episode/back")
def episode_back():
    return session.step_back()


@app.get("/episode/state")
def episode_state():
    if session.index < 0:
        raise HTTPException(400, "No active episode. Call /episode/new first.")
    return session.history[session.index]