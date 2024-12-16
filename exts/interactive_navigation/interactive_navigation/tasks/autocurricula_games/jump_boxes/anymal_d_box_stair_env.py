from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Literal

import matplotlib.pyplot as plt
import os

import interactive_navigation.tasks.autocurricula_games.jump_boxes.mdp as mdp

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise


##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip

ISAAC_GYM_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]

##
# Task-specific configurations
##

from interactive_navigation.tasks.autocurricula_games.jump_boxes.mdp.assets import (
    ROBOT_CFG,
    ROBOT_USD_CFG,
    BOX_CFG,
    TALL_BOX_CFG,
    WALL_CFG,
    SEGMENT_RAY_CASTER_MARKER_CFG,
    HL_RAY_CASTER_MARKER_CFG,
    LL_RAY_CASTER_MARKER_CFG,
)

##
# Scene definition
##
N_BOXES = 1  # number of same boxes

N_STEP_BOXES = 1  # number of different boxes
STEP_HEIGHT = 0.6


# BOXES_DICT = {"short": BOX_CFG, "tall": TALL_BOX_CFG}

# create boxes:
BOXES_DICT = {}

cmap = plt.get_cmap("hsv")
colors = [cmap(i / N_STEP_BOXES) for i in range(N_STEP_BOXES)]
for i in range(N_STEP_BOXES):
    color = colors[i][:3]
    height = STEP_HEIGHT * (i + 1)

    box_prefix = f"box_step_{i + 1}"
    box_prim_path_name = f"Box_{box_prefix}"
    BOXES_DICT[box_prefix] = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"{box_prim_path_name}",
        spawn=sim_utils.CuboidCfg(
            size=(max(height, 1.5), max(height, 1.5), height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
                max_angular_velocity=3.14,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.75, dynamic_friction=0.75, friction_combine_mode="multiply"
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        collision_group=0,
    )


# set step height in terrain generator
terrain_gen_cfg = mdp.terrain.MESH_STEP_TERRAIN_CFG
terrain_gen_cfg.sub_terrains["step"].step_height = STEP_HEIGHT * (N_STEP_BOXES + 1)  # type: ignore


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -- Low level policy sensor
    height_scan_low_level = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
            RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
        ],
        track_mesh_transforms=True,
        visualizer_cfg=LL_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # High level policy sensor
    height_scan_high_level = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 30.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=(5, 5)),
        max_distance=100.0,
        drift_range=(-0.0, 0.0),
        debug_vis=False,
        history_length=0,
        mesh_prim_paths=[
            "/World/ground",
            RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
        ],
        track_mesh_transforms=True,
        visualizer_cfg=HL_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )

    def __post_init__(self):
        for i in range(1, N_BOXES + 1):
            for box_name, box_cfg in BOXES_DICT.items():
                box_strs = [
                    f"{box_name}_{i}",  # entity name
                    box_cfg.prim_path + f"_{i}",  # prim path
                    f"{box_name}_lidar_bot_{i}",  # entity name for the lower lidar sensor
                    f"{box_name}_lidar_top_{i}",  # entity name for the upper lidar sensor
                ]

                # add boxes with lidar sensors (lidar only used for reward computation)
                setattr(self, box_strs[0], box_cfg.replace(prim_path=box_strs[1]))
                setattr(  # lidar 1
                    self,
                    box_strs[2],
                    RayCasterCfg(
                        prim_path=box_strs[1],
                        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                        attach_yaw_only=True,
                        pattern_cfg=patterns.LidarPatternCfg(
                            channels=1,
                            vertical_fov_range=(-0.0, 0.0),
                            horizontal_fov_range=(0, 360),
                            horizontal_res=45,
                        ),
                        max_distance=100.0,
                        debug_vis=False,
                        mesh_prim_paths=["/World/ground"],
                        track_mesh_transforms=False,
                        visualizer_cfg=SEGMENT_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCasterBox"),
                    ),
                )
                setattr(  # lidar 2
                    self,
                    box_strs[3],
                    getattr(self, box_strs[2]).replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0))),
                )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    robot_goal = mdp.GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        randomize_goal=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    interactive_nav_action = mdp.InteractiveNavigationActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(  # copied from velocity_env & box_climb_env
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        locomotion_policy_file=os.path.join(mdp.LOW_LEVEL_NET_PATH, "policy_walk_0310.jit"),
        # "/home/rafael/Projects/MT/interactive_navigation/logs/rsl_rl/anymal_d_rough/locomotion_anymal_d_faster/exported/policy.pt",
        climbing_policy_file=os.path.join(mdp.LOW_LEVEL_NET_PATH, "policy_climb_0310.jit"),
        # "/home/rafael/Projects/MT/interactive_navigation/logs/rsl_rl/anymal_d_ll_box_climb/anymal_d_box_climb_ppo_v2/exported/policy.pt",
        observation_group="low_level_policy",
        locomotion_policy_freq=50.0,
        scale=[0.5, 0.25, 1.0],  # actions = raw_actions * scale + offset, raw_actions squashed to [-1, 1]
        offset=[0.25, 0.0, 0.0],
        debug_vis=True,
        reorder_joint_list=ISAAC_GYM_JOINT_NAMES,
    )

    # usd robot


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # command, 2d pos, sin cos heading, time left [0,1]
        pos_head_time_command = ObsTerm(func=mdp.action_command, params={"action_name": "interactive_nav_action"})

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_low_level_action, params={"action_name": "interactive_nav_action"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scan_low_level")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # self
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        goal_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        height_scan = ObsTerm(
            func=mdp.lidar_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scan_high_level")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-10.0, 10.0),
        )

        # boxes:
        boxes_poses = ObsTerm(
            func=mdp.box_pose,
            params={
                "entity_str": "box",
                "pov_entity": SceneEntityCfg("robot"),
                "return_box_height": N_STEP_BOXES > 1,
                # "return_mask": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


Z_ROBOT = 0.3 + 0.05
Z_BOX = 0.25 + 0.05
Z_WALL = 0.5 + 0.05

first_box_entities = [SceneEntityCfg(box_name + "_1") for box_name in BOXES_DICT.keys()]
other_box_entities = [
    SceneEntityCfg(box_name + f"_{i}") for box_name in BOXES_DICT.keys() for i in range(2, N_BOXES + 1)
]
first_box_entities.reverse()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material_robot = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_box_n_robot = EventTerm(
        func=mdp.reset_boxes_and_robot,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "boxes_sorted": first_box_entities,
            "other_boxes": other_box_entities,
            # "pose_range": {"yaw": (0, 0)},
            "pose_range_robot": {"yaw": (-math.pi / 2, math.pi / 2)},
            "random_dist": True,
            "min_dist": 0.1,
            "robot_z_offset": 0.15,
            "robot_radius": 1.5,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    def __post_init__(self):
        for i in range(1, N_BOXES + 1):
            for box_name in BOXES_DICT.keys():
                box_str = f"{box_name}_{i}"  # entity name
                setattr(
                    self,
                    f"physics_material_{box_str}",
                    EventTerm(
                        func=mdp.randomize_rigid_body_material,  # type: ignore
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(box_str, body_names=".*"),
                            "static_friction_range": (0.6, 1.0),
                            "dynamic_friction_range": (0.4, 0.8),
                            "restitution_range": (0.0, 0.0),
                            "num_buckets": 64,
                        },
                    ),
                )
                setattr(
                    self,
                    f"add_base_mass_{box_str}",
                    EventTerm(
                        func=mdp.randomize_rigid_body_mass,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(box_str, body_names=".*"),
                            "mass_distribution_params": (0.0, 10.0),
                            "operation": "add",
                        },
                    ),
                )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Box interaction
    box_moving = RewTerm(
        func=mdp.BoxMovingReward().box_interaction,
        weight=0.01,
    )

    any_box_close_to_step = RewTerm(
        func=mdp.any_box_close_to_step_reward,
        weight=0.1,
        params={
            "robot_str": "robot",
            "dist_sensor_1_str": "box_short_lidar_bot",
            "dist_sensor_2_str": "box_short_lidar_top",
            "proximity_threshold": 0.5,
            "proximity_std": 0.3,
            "step_size_threshold": 0.75,
        },
    )

    closest_box_close_to_step = RewTerm(
        func=mdp.closest_box_close_to_step_reward,
        weight=0.5,
        params={
            "robot_str": "robot",
            "dist_sensor_1_str": "box_short_lidar_bot",
            "dist_sensor_2_str": "box_short_lidar_top",
            "proximity_threshold": 0.5,
            "proximity_std": 1.0,
            "step_size_threshold": 0.75,
        },
    )

    # TODO: reward for valid stair ie, if first is close to step, second closes to first, etc

    close_to_box = RewTerm(
        func=mdp.CloseToBoxReward().close_to_box_reward,
        weight=0.1,
        params={"threshold": 1.0},
    )

    # Jumping
    successful_jump = RewTerm(
        func=mdp.StepUpReward().successful_jump_reward,
        weight=10,
        params={},  # TODO, this might not work for the anymal robot, since its height may not be constant
    )

    # Moving up
    new_height = RewTerm(
        func=mdp.StepUpReward().new_height_reached_reward,
        weight=200,
        params={},
    )

    # high_up = RewTerm(
    #     func=mdp.high_up,
    #     weight=0.01,
    #     params={"height_range": (0.0, 3.0)},
    # )
    # Moving towards goal
    moving_towards_goal = RewTerm(
        func=mdp.moving_towards_goal,
        weight=1.0,
        params={"command_name": "robot_goal"},
    )

    goal_reached = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
        params={"term_keys": "goal_reached"},
        weight=2500.0,
    )

    # penalty terms
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    too_far = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
        params={"term_keys": "too_far_from_goal"},
        weight=-250.0,
    )
    wasting_time = RewTerm(
        func=mdp.is_alive,
        weight=-0.005,
    )

    flipped = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
        params={"term_keys": "flipped"},
        weight=-200.0,
    )

    time_out = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
        params={"term_keys": "time_out"},
        weight=-100.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out_perturbed, params={"perturbation": 5}, time_out=True)  # type: ignore
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # type: ignore

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
            "goal_cmd_name": "robot_goal",
            "distance_threshold": 1.5,
        },
    )

    too_far_from_goal = DoneTerm(
        func=mdp.too_far_from_goal,
        params={
            "goal_cmd_name": "robot_goal",
            "distance_threshold": 30,
        },
    )

    flipped = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(90)},
    )


DIST_CURR = mdp.DistanceCurriculum(
    start_dist=1.25,
    max_dist=12.0,
    dist_increment=0.1,
    goal_termination_name="goal_reached",
)  # TODO: curriculum where only one random box is moved away from the stair

TERRAIN_CURR = mdp.TerrainCurriculum(
    num_successes=10, num_failures=10, goal_termination_name="goal_reached", random_move_prob=0.05
)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # num_obstacles = CurrTerm(func=mdp.num_boxes_curriculum)

    distance_curriculum = CurrTerm(func=DIST_CURR.entity_entity_dist_curriculum)

    # robot_speed = CurrTerm(
    #     func=mdp.robot_speed_curriculum,
    #     params={"action_term_name": "wrench", "num_steps": 25_000, "start_multiplier": 2.0},
    # )

    # terrain_levels = CurrTerm(func=TERRAIN_CURR.terrain_levels)


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    # eye: tuple[float, float, float] = (-60.0, 0.5, 70.0)
    # eye: tuple[float, float, float] = (2.5, 2.5, 1.0)
    eye: tuple[float, float, float] = (6.0, 6.0, 6.0)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    # lookat: tuple[float, float, float] = (-60.0, 0.0, -10000.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: Literal["world", "env", "asset_root"] = "asset_root"
    """
    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    """
    env_index: int = 0
    asset_name: str | None = "robot"


##
# Environment configuration
##


@configclass
class AnymalBoxeStairEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Data container
    # data_container: mdp.DataContainer = mdp.DataContainer()

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10.0)
    viewer: ViewerCfg = ViewerCfg()

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # -- frequency settings
        self.fz_planner = 10  # 10 Hz
        self.sim.dt = 0.005  # 200 Hz
        self.decimation = int(1 / (self.sim.dt * self.fz_planner))  # 20

        self.episode_length_s = 50.0

        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # GPU settings
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_collision_stack_size = 2**30

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.lidar is not None:
        #     self.scene.lidar.update_period = self.decimation * self.sim.dt
        if self.scene.height_scan_low_level is not None:
            self.scene.height_scan_low_level.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
