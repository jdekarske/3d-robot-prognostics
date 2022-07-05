import numpy as np
from degradation_env import DegradationEnv

# only set in beginning
CONTROL_FREQ = 200
cycle_time = 10
OUTDIR = "./out/"
horizon = CONTROL_FREQ * cycle_time

env = DegradationEnv(
    horizon=horizon,
    has_renderer=False,
    hard_reset=True,
    logging_dir=OUTDIR,
)

filename = "somelabel"  # TODO
cyclenumber = "cycle1"  # TODO
cube_mass = 1  # kg
friction = [500, 1000]
friction_joints = ["robot0_joint2", "robot0_joint1"]

env.cube_mass = cube_mass # must be before the reset
env.reset()
for i, joint in friction_joints:
    env.set_friction(joint, friction[i])
env.run()
