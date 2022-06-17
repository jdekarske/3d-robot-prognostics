import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.controllers import load_controller_config


class DegradationEnv(SingleArmEnv):
    """
    (modified from "lift environment")

    Args:
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
        robot="Panda",
        env_configuration="default",
        # controller_configs=None, # using OSC controller instantiated below
        gripper_types="default",
        initialization_noise=None,  # TODO decide if we want this
        placement_initializer=None,  # TODO randomly place object position
        has_renderer=False,
        has_offscreen_renderer=False,  # TODO probably don't need this
        render_camera=None,  # TODO build a better view and take this out of the args
        render_collision_mesh=False,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,  # TODO convert this to time
        ignore_done=False,  # TODO check the docs for this... very confusing
        hard_reset=True,  # TODO check if we can change degradtion without a hard reset
        camera_names="agentview",  # TODO what is this?
        camera_heights=256,  # TODO decide if this is a good default and remove from args
        camera_widths=256,  # TODO decide if this is a good default and remove from args
    ):
        # unneccessary check so I don't make a stupid mistake later
        if not isinstance(robot, str):
            raise Exception("only use a single robot")

        # Just use the OSC Position controller because it is the easiest
        # TODO check that this controller works with UR5
        controller_config = load_controller_config(default_controller="OSC_POSE")
        controller_config["control_delta"] = False  # actions in world coordinates

        # settings for table top
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0, 0, 0.8))

        # object placement initializer
        self.placement_initializer = placement_initializer

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
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
        )

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

        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
        )  # TODO make this a "space cube"

        # Create placement initializer
        # TODO make this sample based on mass according to "task severity"
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # Set joint attributes
        # TODO design interface to allow us to change damping, friction or other attributes intelligently
        # TODO Test if this can be changed at any time (this suggests you can't for damping
        # TODO look at this method to modify the sim directly: https://github.com/ARISE-Initiative/robosuite/blob/874ce964640f66440a695582a1375df1aff247ac/robosuite/utils/mjmod.py#L1913
        # I think you might be able to modify here: env.mjpy_model.dof_frictionloss
        # https://mujoco.readthedocs.io/en/latest/computation.html#passive-forces)
        # Docs: https://mujoco.readthedocs.io/en/latest/computation.html#friction-loss
        # defaults: https://github.com/ARISE-Initiative/robosuite/blob/874ce964640f66440a695582a1375df1aff247ac/robosuite/models/robots/robot_model.py#L71
        dof = self.robots[0].robot_model.dof
        # self.robots[0].robot_model.set_joint_attribute(
        #     attrib="frictionloss", values=500 * np.ones(dof), force=False
        # )
        # TODO other attributes to change... there are many more! yay!
        self.robots[0].robot_model.set_joint_attribute(
            attrib="damping", values=20 * np.ones(dof), force=False
        )
        # self.robots[0].robot_model.set_joint_attribute(
        #     attrib="armature", values=np.array([5.0 / (i + 1) for i in range(dof)]), force=False
        # )

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

        # TODO add robot EE here

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if
        enabled Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        modality = "object"

        # cube-related observables
        @sensor(modality=modality)
        def cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_body_id])

        @sensor(modality=modality)
        def cube_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

        sensors = [cube_pos, cube_quat]  # TODO add these: gripper_pos, gripper_quat]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that
            specific
                component should be visualized. Should have "grippers" keyword as well as any other
                relevant options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    # TODO not used but might be helpful
    # def _check_success(self):
    #     """
    #     Check if cube has been lifted.
    #     Returns:
    #         bool: True if cube has been lifted
    #     """
    #     cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
    #     table_height = self.model.mujoco_arena.table_offset[2]

    #     # cube is higher than the table top above a margin
    #     return cube_height > table_height + 0.04

    # def step(self, action):
    #     """
    #     TODO set the desired trajectory in this class before the movement so you
    #     can just call ".step" without specifying an action. A trajectory could
    #     look like [(time1,state1), ..., (4,[0.8, 0, 0.8])]. this method would
    #     track the current step and apply the actions as necessary. this saves us
    #     from having to do the `if step > x action = y` kind of thing. feel free
    #     to find a trajectory generator somewhere
    #     """
    #     placeholder_action = action
    #     super().step(placeholder_action)  # this is a placeholder

    def set_trajectory(self, pointlist):
        """
        TODO Process trajectory here include a check for cube orientation so the gripper can match
        the orientation
        """
        raise NotImplementedError()

    def pick_cube(self):
        """
        TODO generate trajectory to pick up the cube
        - move above the cube
        - open gripper
        - move down
        - grab
        -  build a better view and take this out of the args # TODO change this toonvert this to time # TODO check the docs for this... very confusing # TODO check if we can change degradtion withiout thisa hard resetetc
        """
        raise NotImplementedError()

    def reward(self, action):
        return 0  # this is required for some reason
