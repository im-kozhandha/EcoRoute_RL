import sys, os
# Ensure project root (C:\EcoRoute_RL) is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.eco_route_env import EcoRouteEnv

print(sys.executable)
print(sys.path)

import numpy as np
from src.envs.eco_route_env import EcoRouteEnv

def run_random_episode():
    # Create environment with 4 nodes and 3 deliveries
    env = EcoRouteEnv(num_nodes=4, deliveries=[1, 1, 1])

    obs, info = env.reset()
    done = False
    step_count = 0

    print("\n--- Starting Random Episode ---")
    while not done and step_count < 20:
        action = env.action_space.sample()  # pick a random next node
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        print(f"Step {step_count} | Action: {action} | Reward: {reward:.2f}")
        print(f"Obs: Node={obs['current_node']} | Pending={obs['pending']} | "
              f"Dist={obs['distance'][0]:.1f} | Em={obs['emissions'][0]:.2f}")
        print("Info:", info)
        print("-" * 60)

    print("\n--- Episode Finished ---")
    env.render()

if __name__ == "__main__":
    run_random_episode()
