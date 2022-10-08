"""
A sample implementation of the degradation environment.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from degradation_env import DegradationEnv

# only set in beginning
CONTROL_FREQ = 20
CYCLE_TIME = 10
HORIZON = CONTROL_FREQ * CYCLE_TIME
EXPFILE = "inputs/sampleinputs_Exp1.csv"


env = DegradationEnv(
    horizon=HORIZON,
    has_renderer=False,  # set True if you want to see it rendered
    logging_dir="./out/",
    logging_file=EXPFILE + ".hdf5",  # if none uses date and time
    control_freq=CONTROL_FREQ,
)

cycles = pd.read_csv(EXPFILE, header=0)
joints = [item for item in cycles.keys() if "joint" in item]
for _, row in tqdm(cycles.iterrows(), total=len(cycles)):
    env.label = row["label"]
    env.cycle = row["cycle"]
    env.cube_mass = row["payload"]  # kg
    env.final_pos = [-0.1, 0.1, 1.1, np.pi / np.sqrt(2), np.pi / np.sqrt(2), 0, 0.5]

    env.reset()  # MUST BE BEFORE MODYFING SIM PARAMS

    frictions = row[joints][~row[joints].isnull()]
    for joint, val in frictions.items():
        env.set_damping(joint, val)

    env.run()
