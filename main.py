"""
A sample implementation of the degradation environment.
"""

import pandas as pd
from degradation_env import DegradationEnv

# only set in beginning
CONTROL_FREQ = 20
CYCLE_TIME = 10
HORIZON = CONTROL_FREQ * CYCLE_TIME

env = DegradationEnv(
    horizon=HORIZON,
    has_renderer=False, # set True if you want to see it rendered
    logging_dir="./out/",
    logging_file="test.hdf5",  # if none uses date and time
    control_freq=CONTROL_FREQ,
)

cycles = pd.read_csv("./sampleinputs.csv", header=0)
joints = [item for item in cycles.keys() if "joint" in item]
for _, row in cycles.iterrows():
    env.label = row["label"]
    env.cycle = row["cycle"]
    env.cube_mass = row["payload"]  # kg

    frictions = row[joints][~row[joints].isnull()]
    for joint, val in frictions.items():
        env.set_friction(joint, val)

    env.reset()
    env.run()
