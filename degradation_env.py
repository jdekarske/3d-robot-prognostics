"""
For simulating a degrading robot arm by adding friction.
Author: Jason Dekarske (jdekarske@ucdavis.edu)
License: MIT
"""
import os
import datetime
import numpy as np
import h5py

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, mat2euler
from robosuite.controllers import load_controller_config


class DegradationEnv(SingleArmEnv):
    """
    (modified from "lift environment")

    Args:
        logging_dir (str): the directory where output files will be saved.

        robot: Default: Panda has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True.  Setting this value
        to 'None' will result in the default angle being applied, which is useful as it can be
        dragged / panned by the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        Might be useful to see if we are hitting ghost objects

        control_freq (float): how many control signals to receive in every second. This sets the
        amount of simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
        only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single
        str if same name is to be used for all cameras' rendering or else it should be a list of
        cameras to render.  :Note: At least one camera must be specified if @use_camera_obs is True.
        :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"),
        use the convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera
        images from each robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
        same height is to be used for all cameras' frames or else it should be a list of the same
        length as "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
        same width is to be used for all cameras' frames or else it should be a list of the same
        length as "camera names" param.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        logging_dir=None,
        logging_file=None,
        robot="Panda",
        env_configuration="default",
        # controller_configs=None, # using OSC controller instantiated below
        gripper_types="default",
        initialization_noise=None,  # TODO decide if we want this
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_collision_mesh=False,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,  # TODO check if we can change degradtion without a hard reset
    ):
        # unneccessary check so I don't make a stupid mistake later
        if not isinstance(robot, str):
            raise Exception("only use a single robot")

        # Just use the OSC Position controller because it is the easiest
        # TODO check that this controller works with UR5
        controller_config = load_controller_config(default_controller="OSC_POSE")
        controller_config["control_delta"] = False  # actions in world coordinates
        controller_config["interpolation"] = "linear"

        # settings for table top
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0, 0, 0.8))

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.initialization_noise = initialization_noise
        self.cube_body_id = None
        self.cube_mass = 0.0729

        super().__init__(
            robots=robot,
            env_configuration=env_configuration,
            controller_configs=controller_config,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
        )

        self.action = np.zeros(self.action_dim)
        self.start_action = [
            0.1,
            0.0,
            1.0,
            np.pi,
            0.0,
            0.0,
            0.5,
        ]
        self.trajectory = [(self.start_action, 0.0)]

        # set up directory for logging and initial lists
        self.logging_observables = list(self.active_observables)
        self.logging_dir = os.path.abspath(logging_dir)
        if logging_file is None:
            self.logging_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
        else:
            self.logging_file = logging_file
        self.logging = None

        # more parameters for logging
        self.label = None
        self.cycle = None

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Put the robot on the table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        cube_dim = [0.030, 0.030, 0.030]

        # TODO consider a mass sampler
        if not self.initialization_noise:
            cube_min = cube_dim
            cube_max = cube_dim
            placement_range = [0, 0]
        else:
            # TODO the initialization noise should actually affect these
            cube_min = cube_dim - 0.022
            cube_max = cube_dim + 0.022
            placement_range = [-0.03, 0.03]

        avg_vol = np.mean([np.prod(cube_min), np.prod(cube_max)]) # m^3

        self.cube = BoxObject(
            name="cube",
            size_min=cube_min,
            size_max=cube_max,
            rgba=[1, 0, 0, 1],
            density=self.cube_mass / avg_vol, # kg/m^3 2700 kg/m3 for aluminum
        )

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=placement_range,
            y_range=placement_range,
            rotation=0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an index or a list of
        indices that point to the corresponding elements in a flatten array, which is how MuJoCo
        stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if
        enabled Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # this holds robot observations already!
        observables = super()._setup_observables()

        # cube-related observables
        @sensor(modality="object")
        def cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_body_id])

        @sensor(modality="object")
        def cube_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

        @sensor(modality="environment")
        def current_time(obs_cache):
            return self.sim.data.time

        @sensor(modality="proprio")
        def robot0_effort(obs_cache):
            return self.robots[0].torques

        sensors = [cube_pos, cube_quat, current_time, robot0_effort]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, _sensor in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=_sensor,
                sampling_rate=self.control_freq,
            )

        # I want joint angles....
        observables["robot0_joint_pos_cos"].set_active(False)
        observables["robot0_joint_pos_cos"].set_enabled(False)
        observables["robot0_joint_pos_sin"].set_active(False)
        observables["robot0_joint_pos_sin"].set_enabled(False)
        observables["robot0_joint_pos"].set_active(True)
        observables["robot0_joint_pos"].set_enabled(True)

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an
        # xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def set_friction(self, joint_name, val):
        """
        TODO
        https://robosuite.ai/docs/source/robosuite.utils.html?highlight=density#robosuite.utils.mjmod.DynamicsModder
        Available "joint" names = ('robot0_joint1', 'robot0_joint2', 'robot0_joint3',
        'robot0_joint4', 'robot0_joint5', 'robot0_joint6', 'robot0_joint7',
        'gripper0_finger_joint1', 'gripper0_finger_joint2', 'cube_joint0')
        """
        jnt_id = self.sim.model.joint_name2id(joint_name)
        if self.sim.model.jnt_type[jnt_id] != 0:
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            self.sim.model.dof_frictionloss[dof_idx] = val

    # Other potential modifiers:
    # actuator_ctrlrange
    # actuator_forcelimited
    # dof_damping
    # sensor_names (gripper force)

    def step(self, action=None, use_trajectory=True):
        """
        TODO improve linear trajectory generator
        """

        if self.timestep == 0:
            cube_ori = mat2euler(self.sim.data.body_xmat[self.cube_body_id].reshape(3, 3))
            self.set_trajectory(self.sim.data.body_xpos[self.cube_body_id], cube_ori[2])

        # sample the trajectory
        if use_trajectory:
            if self.trajectory:
                if self.cur_time >= self.trajectory[0][1]:
                    self.action = self.trajectory.pop(0)[0]
        else:
            self.action = action

        obs, reward, done, info = super().step(np.array(self.action))

        obs_conc = np.concatenate([np.atleast_1d(obs[x]) for x in self.logging_observables])

        # initialize the logging object if necessary
        if (self.logging is None) and (self.logging_dir is not None):
            self.logging = np.empty((self.horizon, obs_conc.size))
        self.logging[self.timestep - 1][:] = obs_conc
        if done:
            self.save_data()
        return obs, reward, done, info

    def set_trajectory(self, cube_pos, cube_yaw):
        """
        TODO Process trajectory here include a check for cube orientation so the gripper can match
        the orientation
        """

        # manually add some trajectories to pick up the cube
        # TODO can't figure out how to change EE yaw
        # I think it has to do with *rot_offset
        # for i in range(20):
        #     self.trajectory.append(
        #         ([cube_pos[0], self.start_action[1], self.start_action[2], np.pi, -np.pi*i/20,0.0,  0], i+5)
        #     )

        self.trajectory.append(
            ([cube_pos[0], cube_pos[1], cube_pos[2] + 0.05, np.pi, 0.0, 0.0, -0.9], 2)
        )
        self.trajectory.append(([cube_pos[0], cube_pos[1], cube_pos[2], np.pi, 0.0, 0.0, -0.9], 4))
        self.trajectory.append(([cube_pos[0], cube_pos[1], cube_pos[2], np.pi, 0.0, 0.0, 0.5], 6))
        self.trajectory.append(
            ([cube_pos[0], cube_pos[1], cube_pos[2] + 0.2, np.pi, 0.0, 0.0, 0.5], 8)
        )

    def reward(self, action):
        return 0  # this is required for some reason

    def run(self):
        """
        This will go through all the steps that are probably necessary. Note that this is blocking
        and will take a while
        """
        if self.logging_dir is not None:
            if self.label is None:
                raise Exception("you must set a label property before running")
            if self.cycle is None:
                raise Exception("you must set a cycle property before running")

        while not self.done:
            self.step(self.action)  # take action in the environment
            if self.has_renderer:
                self.render()  # render on display

        self.label = None
        self.cycle = None

    def save_data(self):
        """
        Output data to a file
        """
        if not self.logging_dir:
            return
        if not os.path.exists(self.logging_dir):
            os.mkdir(self.logging_dir)
        with h5py.File(os.path.join(self.logging_dir, self.logging_file), "a") as _file:
            # If the label exists, append the cycle
            if self.label in _file.keys():
                self.label += str(self.cycle)
            _file.create_dataset(self.label, data=self.logging)
            _file.attrs["header"] = self.logging_observables
            _file.attrs["header_dim"] = [
                self.observation_spec()[x].size for x in self.logging_observables
            ]
