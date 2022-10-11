"""
The loop that processes an inputs csv file
"""

import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from degradation_env import DegradationEnv

# defaults
CONTROL_FREQ = 1000
CYCLE_TIME = 5
HORIZON = CONTROL_FREQ * CYCLE_TIME
FINAL_POS = [-0.1, 0.1, 1.1]


def experiment(
    experimentfile,
    experimentfileout,
    horizon=HORIZON,
    control_freq=CONTROL_FREQ,
    final_pos=FINAL_POS,
):
    env = DegradationEnv(
        horizon=horizon,
        logging_dir="./out/",
        logging_file=experimentfileout,
        control_freq=control_freq,
        compression="gzip",
    )

    cycles = pd.read_csv(experimentfile, header=0)
    joints = [item for item in cycles.keys() if "joint" in item]
    for _, row in tqdm(cycles.iterrows(), total=len(cycles)):  # todo multiprocessing
        env.label = row["label"]
        env.cycle = row["cycle"]
        env.cube_mass = row["payload"]  # kg
        env.final_pos = np.hstack((final_pos, [np.pi / np.sqrt(2), np.pi / np.sqrt(2), 0, 0.5]))
        env.reset()  # MUST BE BEFORE MODYFING SIM PARAMS

        frictions = row[joints][~row[joints].isnull()]
        for joint, val in frictions.items():
            env.set_damping(joint, val)

        env.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run the prognostic simulation.")

    parser.add_argument("--input_file", metavar="-i", type=str)
    parser.add_argument("--output_file", metavar="-o", type=str)
    parser.add_argument("--cycle_time", metavar="-y", type=int, default=CYCLE_TIME)
    parser.add_argument("--control_freq", metavar="-c", type=int, default=CONTROL_FREQ)
    parser.add_argument("--final_pos", metavar="-p", nargs=3, type=list, default=FINAL_POS)
    parser.add_argument("--test", action="store_true")

    args = vars(parser.parse_args())

    if args["test"]:
        args["input_file"] = "sampleinputs.csv"
        args["output_file"] = "test.hdf5"
        print(os.getcwd())
        if os.path.exists("out/" + args["output_file"]):
            print("test file exists, removing...")
            os.remove("out/" + args["output_file"])

    if not args["output_file"]:
        args["output_file"] = (
            args["input_file"].split("/")[-1].split(".")[0] + ".hdf5"
        )  # TODO use os path

    experiment(
        args["input_file"],
        args["output_file"],
        horizon=args["control_freq"] * args["cycle_time"],
        control_freq=args["control_freq"],
        final_pos=args["final_pos"],
    )
