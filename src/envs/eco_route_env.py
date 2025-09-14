import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.utils import build_graph, calculate_emissions

class EcoRouteEnv(gym.Env):
    """Custom RL environment for eco-friendly vehicle routing."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_nodes=4, deliveries=None):
        super(EcoRouteEnv, self).__init__()

        # Build graph (adjacency, distances)
        self.graph, self.distances = build_graph(num_nodes)

        # Define deliveries (default = one per node except depot)
        if deliveries is None:
            deliveries = [1] * (num_nodes - 1)  # e.g. 3 deliveries if 4 nodes
        self.initial_deliveries = np.array(deliveries, dtype=int)

        # State = (current_node, pending_deliveries, cumulative_distance, cumulative_emissions)
        self.num_nodes = num_nodes
        self.state = None

        # Action space: choose next node (discrete over all nodes)
        self.action_space = spaces.Discrete(self.num_nodes)

        # Observation space: deliveries + current node + dist/emissions
        self.observation_space = spaces.Dict({
            "current_node": spaces.Discrete(self.num_nodes),
            "pending": spaces.MultiBinary(self.num_nodes - 1),
            "distance": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "emissions": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = {
            "current_node": 0,  # depot
            "pending": self.initial_deliveries.copy(),
            "distance": np.array([0.0], dtype=np.float32),
            "emissions": np.array([0.0], dtype=np.float32),
        }
        return self.state, {}

    def step(self, action):
        prev_node = self.state["current_node"]
        next_node = action

        # Distance traveled
        dist = self.distances[prev_node][next_node]
        self.state["distance"] += dist

        # Emissions
        emissions = calculate_emissions(dist)
        self.state["emissions"] += emissions

        # Update deliveries
        if next_node > 0 and self.state["pending"][next_node - 1] == 1:
            self.state["pending"][next_node - 1] = 0

        # Update current node
        self.state["current_node"] = next_node

        # Check if done (all deliveries completed)
        done = np.all(self.state["pending"] == 0)

        # Reward (negative distance + emissions)
        reward = -(dist + emissions)

        info = {
            "distance": float(self.state["distance"]),
            "emissions": float(self.state["emissions"]),
            "deliveries_left": self.state["pending"].sum(),
        }

        print(f"At node {next_node}, Pending: {self.state['pending']}, "
              f"Distance: {self.state['distance'][0]}, Emissions: {self.state['emissions'][0]}")

        return self.state, reward, done, False, info

    def render(self):
        from src.envs.visualization import plot_graph
        plot_graph(self.graph, self.state)
