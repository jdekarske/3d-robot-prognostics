import numpy as np
from degradation_env import DegradationEnv

OUTDIR = "./out"

env = DegradationEnv(
    horizon=100, has_renderer=False, hard_reset=True, logging_dir=OUTDIR
)
env.set_friction("robot0_joint1", 1000)
env.reset()

env.run()
