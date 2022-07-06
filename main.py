from degradation_env import DegradationEnv

# only set in beginning
CONTROL_FREQ = 20
CYCLE_TIME = 10
OUTDIR = "./out/"
horizon = CONTROL_FREQ * CYCLE_TIME

env = DegradationEnv(
    horizon=horizon,
    has_renderer=True,
    hard_reset=True,
    logging_dir=OUTDIR,
    control_freq=CONTROL_FREQ
)

filename = "somelabel"  # TODO
cyclenumber = "cycle1"  # TODO
cube_mass = 1  # kg
friction = [500, 1000]
friction_joints = ["robot0_joint2", "robot0_joint1"]

env.cube_mass = cube_mass  # must be before the reset
env.reset()
# for i, joint in enumerate(friction_joints):
# env.set_friction(joint, friction[i])
env.run()
