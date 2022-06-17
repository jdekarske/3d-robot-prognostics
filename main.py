import numpy as np
from degradation_env import DegradationEnv

env = DegradationEnv(has_renderer=True, hard_reset=True)

env.reset()

action = [0.1, 0.0, 1.0, np.pi, 0.0, 0.0, 0.5]

for i in range(1000):
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

# env.model.mujoco_robots[0]
