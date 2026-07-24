"""
Run this once before launching app.py:

    python train_all.py

Trains Q-learning, SARSA, and DQN, then saves each to saved_models/
so the UI can load them instantly instead of retraining every launch.

Defaults to 2000 episodes each for a reasonably quick run (a few
minutes on a laptop CPU). Bump NUM_EPISODES up for closer-to-final
results, matching the 5000 used in the original notebook.
"""

import pickle
import torch

from agents import train_q_learning, train_sarsa, train_dqn

NUM_EPISODES = 5000

if __name__ == "__main__":

    print("Training Q-learning...")
    Q_table, q_rewards, q_queues = train_q_learning(num_episodes=NUM_EPISODES)
    with open("saved_models/q_table.pkl", "wb") as f:
        pickle.dump(dict(Q_table), f)
    with open("saved_models/q_history.pkl", "wb") as f:
        pickle.dump({"rewards": q_rewards, "queues": q_queues}, f)

    print("\nTraining SARSA...")
    Q_table_sarsa, sarsa_rewards, sarsa_queues = train_sarsa(num_episodes=NUM_EPISODES)
    with open("saved_models/sarsa_table.pkl", "wb") as f:
        pickle.dump(dict(Q_table_sarsa), f)
    with open("saved_models/sarsa_history.pkl", "wb") as f:
        pickle.dump({"rewards": sarsa_rewards, "queues": sarsa_queues}, f)

    print("\nTraining DQN...")
    dqn_model, dqn_rewards, dqn_queues, dqn_losses = train_dqn(num_episodes=NUM_EPISODES)
    torch.save(dqn_model.state_dict(), "saved_models/dqn.pt")
    with open("saved_models/dqn_history.pkl", "wb") as f:
        pickle.dump({"rewards": dqn_rewards, "queues": dqn_queues}, f)

    print("\nAll models trained and saved to saved_models/")
