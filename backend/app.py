"""
Run after train_all.py has produced files in saved_models/:

    python app.py

Opens a local Gradio app (prints a URL like http://127.0.0.1:7860).
"""

import pickle
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt

from environment import Traffic_Four_Way_Intersection_Environment
from discretize import discretize_state
from agents import DQN
from visualize import plot_intersection_state

MAX_QUEUE = 20
NUM_BUCKETS = 2
NUM_ACTIONS = 12

# -------------------------------------------------------------
# LOAD TRAINED MODELS
# -------------------------------------------------------------

def load_table(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_history(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    Q_TABLE = load_table("saved_models/q_table.pkl")
    SARSA_TABLE = load_table("saved_models/sarsa_table.pkl")
    DQN_MODEL = DQN(state_size=16, action_size=NUM_ACTIONS)
    DQN_MODEL.load_state_dict(torch.load("saved_models/dqn.pt"))
    DQN_MODEL.eval()

    Q_HISTORY = load_history("saved_models/q_history.pkl")
    SARSA_HISTORY = load_history("saved_models/sarsa_history.pkl")
    DQN_HISTORY = load_history("saved_models/dqn_history.pkl")
    MODELS_LOADED = True
except FileNotFoundError:
    MODELS_LOADED = False


def pick_action(algorithm, state):
    """Returns the action index chosen by the given trained agent (no exploration)."""
    if algorithm == "Q-learning":
        key = discretize_state(state, MAX_QUEUE, NUM_BUCKETS)
        values = Q_TABLE.get(key, np.zeros(NUM_ACTIONS))
        return int(np.argmax(values))

    if algorithm == "SARSA":
        key = discretize_state(state, MAX_QUEUE, NUM_BUCKETS)
        values = SARSA_TABLE.get(key, np.zeros(NUM_ACTIONS))
        return int(np.argmax(values))

    if algorithm == "DQN":
        with torch.no_grad():
            state_t = torch.tensor(state / MAX_QUEUE, dtype=torch.float32).unsqueeze(0)
            return int(DQN_MODEL(state_t).argmax().item())

    # Baseline: always serve the largest incoming queue
    return int(np.argmax(state[:NUM_ACTIONS]))


MOVEMENT_NAMES = [
    "N_W", "N_S", "N_E", "S_W", "S_N", "S_E",
    "E_N", "E_S", "E_W", "W_N", "W_S", "W_E"
]


# -------------------------------------------------------------
# LIVE SIMULATION TAB CALLBACKS
# -------------------------------------------------------------

def new_episode(algorithm):
    env = Traffic_Four_Way_Intersection_Environment(use_argmax_policy=False)
    state, _ = env.reset()
    fig = plot_intersection_state(state, title=f"{algorithm} -- step 0")
    info_text = "Episode started. Click 'Step' to advance."
    return env, state, fig, info_text


def step_once(algorithm, env, state):
    if env is None:
        return gr.update(), gr.update(), gr.update(), "Click 'New episode' first."

    action = pick_action(algorithm, state)
    next_state, reward, terminated, truncated, info = env.step(action)

    fig = plot_intersection_state(
        next_state,
        selected_movement=info["selected_movement"],
        title=f"{algorithm} -- step {env.current_step}"
    )
    info_text = (
        f"Green light: {info['selected_movement']}  |  "
        f"Cars passed: {info['cars_passed']}  |  "
        f"Total system queue: {info['total_system_queue']}  |  "
        f"Reward: {reward:.0f}"
    )
    if truncated:
        info_text += "\n\nEpisode finished -- click 'New episode' to restart."

    return env, next_state, fig, info_text


# -------------------------------------------------------------
# COMPARISON DASHBOARD TAB
# -------------------------------------------------------------

def build_comparison_charts():
    if not MODELS_LOADED:
        return None, None

    window = 100
    labels = ["Q-learning", "SARSA", "DQN"]
    histories = [Q_HISTORY, SARSA_HISTORY, DQN_HISTORY]

    avg_rewards = [np.mean(h["rewards"][-window:]) for h in histories]
    avg_queues = [np.mean(h["queues"][-window:]) for h in histories]

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(labels, avg_rewards, color=["#1D9E75", "#D85A30", "#7F77DD"])
    ax1.set_title(f"Average reward (last {window} episodes)")
    ax1.set_ylabel("Reward (less negative = better)")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(labels, avg_queues, color=["#1D9E75", "#D85A30", "#7F77DD"])
    ax2.set_title(f"Average total queue (last {window} episodes)")
    ax2.set_ylabel("Cars waiting (lower = better)")

    return fig1, fig2


# -------------------------------------------------------------
# GRADIO APP
# -------------------------------------------------------------

with gr.Blocks(title="Traffic-Signal RL Demo") as demo:
    gr.Markdown("# Adaptive traffic-signal reinforcement learning")

    if not MODELS_LOADED:
        gr.Markdown(
            "**No trained models found in `saved_models/`.** "
            "Run `python train_all.py` first, then restart this app."
        )

    with gr.Tab("Live simulation"):
        with gr.Row():
            algo_dropdown = gr.Dropdown(
                choices=["Baseline", "Q-learning", "SARSA", "DQN"],
                value="Q-learning",
                label="Controller"
            )
            new_ep_btn = gr.Button("New episode")
            step_btn = gr.Button("Step")

        sim_plot = gr.Plot()
        sim_info = gr.Textbox(label="Step details", interactive=False)

        env_state = gr.State(None)
        queue_state = gr.State(None)

        new_ep_btn.click(
            new_episode,
            inputs=[algo_dropdown],
            outputs=[env_state, queue_state, sim_plot, sim_info]
        )
        step_btn.click(
            step_once,
            inputs=[algo_dropdown, env_state, queue_state],
            outputs=[env_state, queue_state, sim_plot, sim_info]
        )

    with gr.Tab("Comparison dashboard"):
        gr.Markdown(
            "Average performance over each agent's final training episodes "
            "(from `train_all.py`). For a truly fair head-to-head, evaluate "
            "all three with exploration off on identical seeds, as the "
            "original notebook does in Step 8."
        )
        refresh_btn = gr.Button("Load results")
        reward_plot = gr.Plot()
        queue_plot = gr.Plot()

        refresh_btn.click(build_comparison_charts, outputs=[reward_plot, queue_plot])

if __name__ == "__main__":
    demo.launch()
