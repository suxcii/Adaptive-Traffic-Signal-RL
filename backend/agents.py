import numpy as np
import random
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim

from environment import Traffic_Four_Way_Intersection_Environment
from discretize import discretize_state

# Hyperparameters matching the notebook's canonical trainers
# (train_tabular_final / train_dqn_final, Step 7 -> Step 8b):
# Q-learning / SARSA: alpha=0.1, gamma=0.9
# DQN: gamma=0.9, lr=0.001, loss=SmoothL1Loss (Huber), min_replay_size=500


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================
# Q-LEARNING
# =============================================================

def train_q_learning(
    num_episodes=5000,
    max_queue_length=20,
    num_buckets=2,
    green_light_capacity=5,
    outgoing_road_capacity=5,
    max_steps=100,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    seed=42
):
    set_global_seed(seed)
    env = Traffic_Four_Way_Intersection_Environment(
        max_queue_length=max_queue_length,
        green_light_capacity=green_light_capacity,
        outgoing_road_capacity=outgoing_road_capacity,
        max_steps=max_steps,
        use_argmax_policy=False
    )
    num_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(num_actions))

    epsilon = epsilon_start
    episode_rewards, episode_queue_totals = [], []

    for episode in range(num_episodes):
        state, info = env.reset()
        state_key = discretize_state(state, max_queue_length, num_buckets)
        total_reward, total_queue = 0, 0
        terminated = truncated = False

        while not (terminated or truncated):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state_key]))

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_key = discretize_state(next_state, max_queue_length, num_buckets)

            done = terminated or truncated
            best_next_value = 0.0 if done else np.max(Q[next_state_key])
            td_target = reward + gamma * best_next_value
            td_error = td_target - Q[state_key][action]
            Q[state_key][action] += alpha * td_error

            state_key = next_state_key
            total_reward += reward
            total_queue += info["total_system_queue"]

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_queue_totals.append(total_queue)

        if (episode + 1) % max(1, num_episodes // 10) == 0:
            print(f"[Q-learning] episode {episode+1}/{num_episodes} | epsilon={epsilon:.3f}")

    return Q, episode_rewards, episode_queue_totals


# =============================================================
# SARSA
# =============================================================

def train_sarsa(
    num_episodes=5000,
    max_queue_length=20,
    num_buckets=2,
    green_light_capacity=5,
    outgoing_road_capacity=5,
    max_steps=100,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    seed=42
):
    set_global_seed(seed)
    env = Traffic_Four_Way_Intersection_Environment(
        max_queue_length=max_queue_length,
        green_light_capacity=green_light_capacity,
        outgoing_road_capacity=outgoing_road_capacity,
        max_steps=max_steps,
        use_argmax_policy=False
    )
    num_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(num_actions))

    epsilon = epsilon_start
    episode_rewards, episode_queue_totals = [], []

    def choose_action(state_key, epsilon):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return int(np.argmax(Q[state_key]))

    for episode in range(num_episodes):
        state, info = env.reset()
        state_key = discretize_state(state, max_queue_length, num_buckets)
        action = choose_action(state_key, epsilon)
        total_reward, total_queue = 0, 0
        terminated = truncated = False

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_key = discretize_state(next_state, max_queue_length, num_buckets)
            done = terminated or truncated

            if not done:
                next_action = choose_action(next_state_key, epsilon)
                td_target = reward + gamma * Q[next_state_key][next_action]
            else:
                next_action = None
                td_target = reward

            td_error = td_target - Q[state_key][action]
            Q[state_key][action] += alpha * td_error

            state_key = next_state_key
            if not done:
                action = next_action

            total_reward += reward
            total_queue += info["total_system_queue"]

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_queue_totals.append(total_queue)

        if (episode + 1) % max(1, num_episodes // 10) == 0:
            print(f"[SARSA] episode {episode+1}/{num_episodes} | epsilon={epsilon:.3f}")

    return Q, episode_rewards, episode_queue_totals


# =============================================================
# DQN
# =============================================================

class DQN(nn.Module):
    def __init__(self, state_size=16, action_size=12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    num_episodes=5000,
    max_queue_length=20,
    green_light_capacity=5,
    outgoing_road_capacity=5,
    max_steps=100,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    lr=0.001,
    seed=42,
    batch_size=64,
    target_update_freq=10,
    replay_buffer_capacity=50000,
    minimum_replay_size=500
):
    set_global_seed(seed)
    env = Traffic_Four_Way_Intersection_Environment(
        max_queue_length=max_queue_length,
        green_light_capacity=green_light_capacity,
        outgoing_road_capacity=outgoing_road_capacity,
        max_steps=max_steps,
        use_argmax_policy=False
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # Huber loss -- matches the notebook's canonical DQN trainer
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    epsilon = epsilon_start
    episode_rewards, episode_queue_totals, losses = [], [], []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state / max_queue_length, dtype=torch.float32).unsqueeze(0)
        total_reward, total_queue = 0, 0
        terminated = truncated = False

        while not (terminated or truncated):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_t = torch.tensor(next_state / max_queue_length, dtype=torch.float32).unsqueeze(0)
            done = terminated or truncated

            replay_buffer.add(
                state.squeeze(0).numpy(), action, reward,
                next_state_t.squeeze(0).numpy(), done
            )

            state = next_state_t
            total_reward += reward
            total_queue += info["total_system_queue"]

            if len(replay_buffer) > max(batch_size, minimum_replay_size):
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                current_q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + (1 - dones) * gamma * next_q_values

                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_queue_totals.append(total_queue)

        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % max(1, num_episodes // 10) == 0:
            print(f"[DQN] episode {episode+1}/{num_episodes} | epsilon={epsilon:.3f}")

    return policy_net, episode_rewards, episode_queue_totals, losses