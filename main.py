import numpy as np
from degradation_env import DegradationEnv

OUTDIR = "./out/"
filename = "somelabel"
cyclenumber = "cycle1"
cube_start_position = [0,0,0]
cube_mass = 1 # kg 
friction = [500,1000]
friction_joints = ["robot0_joint2","robot0_joint1"]
cycle_time = 10

env = DegradationEnv(
    horizon=100, has_renderer=False, hard_reset=True, logging_dir=OUTDIR
)
env.reset()

env.run()

env.reset()
env.set_friction("robot0_joint1", 1000)
env.run()
