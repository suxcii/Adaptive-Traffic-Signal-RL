import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Traffic_Four_Way_Intersection_Environment(gym.Env):
    """
    A four-way traffic intersection.

    12 incoming movement queues (e.g. N_W = arrived from North, wants West)
    + 4 outgoing road queues (N_N, S_S, E_E, W_W) = 16 total queue values.

    The agent picks 1 of 12 incoming movements to give a green light to
    each step. Reward = -(total cars waiting anywhere in the system).
    """

    def __init__(
        self,
        episodes=5000,
        max_queue_length=20,
        green_light_capacity=5,
        outgoing_road_capacity=5,
        max_steps=100,
        use_argmax_policy=False,
        use_sensor_noise=False,
        sensor_noise_range=1,
        use_blocked_penalty=False,
        blocked_penalty_weight=1.0,
        use_fairness_penalty=False,
        fairness_penalty_weight=0.1
    ):
        self.movements = [
            "N_W", "N_S", "N_E",
            "S_W", "S_N", "S_E",
            "E_N", "E_S", "E_W",
            "W_N", "W_S", "W_E"
        ]
        self.num_movements = len(self.movements)

        self.outgoing_movements = ["N_N", "S_S", "E_E", "W_W"]
        self.num_outgoing = len(self.outgoing_movements)

        self.queue_names = self.movements + self.outgoing_movements
        self.num_total_queues = len(self.queue_names)

        self.action_space = spaces.Discrete(self.num_movements)
        self.observation_space = spaces.MultiDiscrete(
            [max_queue_length + 1] * self.num_total_queues
        )

        self.max_queue = max_queue_length
        self.max_steps = max_steps
        self.service_capacity = green_light_capacity
        self.outgoing_road_capacity = outgoing_road_capacity

        self.movement_to_outgoing = {
            "N_W": 15, "S_W": 15, "E_W": 15,   # -> W_W
            "N_S": 13, "E_S": 13, "W_S": 13,   # -> S_S
            "N_E": 14, "S_E": 14, "W_E": 14,   # -> E_E
            "S_N": 12, "E_N": 12, "W_N": 12    # -> N_N
        }

        self.use_argmax_policy = use_argmax_policy
        self.use_sensor_noise = use_sensor_noise
        self.sensor_noise_range = sensor_noise_range

        self.use_blocked_penalty = use_blocked_penalty
        self.blocked_penalty_weight = blocked_penalty_weight
        self.use_fairness_penalty = use_fairness_penalty
        self.fairness_penalty_weight = fairness_penalty_weight
        self.steps_since_served = np.zeros(self.num_movements, dtype=int)

        self.state = None
        self.current_step = 0

    def sensor_reading(self, true_state):
        noise = self.np_random.integers(
            low=-self.sensor_noise_range,
            high=self.sensor_noise_range + 1,
            size=self.num_total_queues
        )
        observed = true_state + noise
        observed = np.clip(observed, 0, self.max_queue)
        return observed

    def _get_observation(self):
        if self.use_sensor_noise:
            return self.sensor_reading(self.state)
        return self.state.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_since_served = np.zeros(self.num_movements, dtype=int)

        incoming_state = self.np_random.integers(low=0, high=6, size=self.num_movements)
        outgoing_state = np.zeros(self.num_outgoing, dtype=int)
        self.state = np.concatenate((incoming_state, outgoing_state))

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        state_before = self.state.copy()
        next_state = self.state.copy()

        # cars already on outgoing roads leave the system
        outgoing_departures = np.minimum(next_state[12:16], self.outgoing_road_capacity)
        next_state[12:16] -= outgoing_departures

        # decide which movement gets the green light
        if self.use_argmax_policy:
            selected_index = int(np.argmax(next_state[:self.num_movements]))
        else:
            if not self.action_space.contains(action):
                raise ValueError(
                    f"Invalid action {action}. Must be 0 to {self.num_movements - 1}."
                )
            selected_index = int(action)

        selected_movement = self.movements[selected_index]

        self.steps_since_served += 1
        self.steps_since_served[selected_index] = 0

        outgoing_index = self.movement_to_outgoing[selected_movement]
        outgoing_movement = self.queue_names[outgoing_index]

        cars_waiting = next_state[selected_index]
        requested_to_pass = min(cars_waiting, self.service_capacity)
        outgoing_space_available = self.max_queue - next_state[outgoing_index]
        cars_passed = min(requested_to_pass, outgoing_space_available)

        next_state[selected_index] -= cars_passed
        next_state[outgoing_index] += cars_passed
        blocked_by_outgoing = requested_to_pass - cars_passed

        # new cars arrive
        arrivals = self.np_random.integers(low=0, high=3, size=self.num_movements)
        blocked_arrivals = 0
        for movement_index in range(self.num_movements):
            for _ in range(arrivals[movement_index]):
                if next_state[movement_index] < self.max_queue:
                    next_state[movement_index] += 1
                else:
                    blocked_arrivals += 1

        incoming_waiting = int(np.sum(next_state[:self.num_movements]))
        outgoing_waiting = int(np.sum(next_state[self.num_movements:]))
        total_system_queue = incoming_waiting + outgoing_waiting
        reward = -total_system_queue

        if self.use_blocked_penalty:
            reward -= self.blocked_penalty_weight * (blocked_by_outgoing + blocked_arrivals)

        if self.use_fairness_penalty:
            reward -= self.fairness_penalty_weight * np.sum(self.steps_since_served)

        self.state = next_state
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "selected_movement": selected_movement,
            "selected_index": selected_index,
            "outgoing_movement": outgoing_movement,
            "outgoing_index": outgoing_index,
            "selected_queue_before_green": int(cars_waiting),
            "cars_passed": int(cars_passed),
            "blocked_by_outgoing": int(blocked_by_outgoing),
            "blocked_arrivals": int(blocked_arrivals),
            "incoming_waiting": incoming_waiting,
            "outgoing_waiting": outgoing_waiting,
            "total_system_queue": total_system_queue,
            "cars_left_system": int(np.sum(outgoing_departures)),
            "steps_since_served": self.steps_since_served.copy(),
            # -- new: raw per-step deltas, for the details panel --
            "state_before": state_before.tolist(),
            "state_after": next_state.tolist(),
            "arrivals": arrivals.tolist(),
            "outgoing_departures": outgoing_departures.tolist(),
        }

        return self._get_observation(), reward, terminated, truncated, info
