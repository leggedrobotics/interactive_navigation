from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Literal

import interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp as mdp

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
from omni.isaac.lab_assets.ant import ANT_CFG
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip


##
# Task-specific configurations
##

from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.assets import (
    ROBOT_CFG,
    ROBOT_USD_CFG,
    CUBOID_FLAT_CFG,
    CUBOID_BIG_CFG,
    CUBOID_SMALL_CFG,
    CUBOID_TALL_CFG,
    WALL_CFG,
    SEGMENT_RAY_CASTER_MARKER_CFG,
)

##
# Scene definition
##
N_BOXES = 1


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=mdp.terrain.PYRAMID_TERRAINS_CFG,
        max_init_terrain_level=500,
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

    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
            RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
        ],
        track_mesh_transforms=True,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )

    # box
    box1 = CUBOID_FLAT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_1",
        init_state=CUBOID_FLAT_CFG.InitialStateCfg(pos=[0.0, 0.75, 0.25]),
    )
    # box2 = CUBOID_BIG_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Box_2",
    #     init_state=CUBOID_BIG_CFG.InitialStateCfg(pos=[0.0, -2.0, 0.5]),
    # )
    # box3 = CUBOID_SMALL_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Box_3",
    #     init_state=CUBOID_SMALL_CFG.InitialStateCfg(pos=[1.5, 0.0, 0.25]),
    # )
    # box4 = CUBOID_TALL_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Box_4",
    #     init_state=CUBOID_TALL_CFG.InitialStateCfg(pos=[-1.5, 0.0, 0.5]),
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy.
        These observations need to be available from the robot's perspective.
        """

        origin = ObsTerm(
            func=mdp.origin_b,  # velocity_2d_b, rotation_velocity_2d_b
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        # my_pose = ObsTerm(
        #     func=mdp.pose_3d_env,  # velocity_2d_b, rotation_velocity_2d_b
        #     params={"entity_cfg": SceneEntityCfg("robot")},
        # )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # box_pose = ObsTerm(
        #     func=mdp.pose_3d_env,
        #     params={
        #         "entity_cfg": SceneEntityCfg("box"),
        #     },
        # )

        box_pose = ObsTerm(
            func=mdp.box_pose_3d,
            params={
                "entity_str": "box",
                "pov_entity": SceneEntityCfg("robot"),
            },
        )

        # proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class MetraStateCfg(ObsGroup):
        """Observations for the metra."""

        # # self
        # # my_pose = ObsTerm(
        # #     func=mdp.pose_3d_env,  # velocity_2d_b, rotation_velocity_2d_b
        # #     params={"entity_cfg": SceneEntityCfg("robot")},
        # # )
        # # box_pose = ObsTerm(
        # #     func=mdp.pose_3d_env,
        # #     params={
        # #         "entity_cfg": SceneEntityCfg("box"),
        # #     },
        # # )
        # # my_velocity = ObsTerm(
        # #     func=mdp.velocity_3d_w,  # velocity_2d_b, rotation_velocity_2d_b
        # #     params={"entity_cfg": SceneEntityCfg("robot")},
        # # )

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        my_pose = ObsTerm(
            func=mdp.origin_b,  # velocity_2d_b, rotation_velocity_2d_b
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        box_pose = ObsTerm(
            func=mdp.box_pose_3d,
            params={
                "entity_str": "box",
                "pov_entity": SceneEntityCfg("robot"),
            },
        )

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class InstructorObsCfg(ObsGroup):
        """Observations for the style instructor group."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    metra: MetraStateCfg = MetraStateCfg()  # currently not used
    instructor: InstructorObsCfg = InstructorObsCfg()


reset_value = 0.1
reset_value_pos = 0.05


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-reset_value_pos, reset_value_pos),
                "y": (-reset_value_pos - 0.75, reset_value_pos - 0.75),
                # "z": (0.35, 0.35),
                # "yaw": (-0.1, 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-reset_value, reset_value),
                "y": (-reset_value, reset_value),
                "z": (-reset_value, reset_value),
                "roll": (-reset_value, reset_value),
                "pitch": (-reset_value, reset_value),
                "yaw": (-reset_value, reset_value),
            },
        },
    )

    reset_box1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box1"),
        },
    )
    # reset_box2 = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("box2"),
    #     },
    # )
    # reset_box3 = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("box3"),
    #     },
    # )
    # reset_box4 = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("box4"),
    #     },
    # )

    # reset_robot_fix = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={"pose_range": {"x": (11.0, 11.0), "y": (11.0, 11.0), "z": (Z_ROBOT, Z_ROBOT)}, "velocity_range": {}},
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # reset_robot_yaw_joint = EventTerm(
    #     func=mdp.reset_id_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-3.1416, 3.1416),  # (-3.1416, 3.1416),
    #         "velocity_range": (0.0, 0.0),
    #         "joint_names": ["joint_yaw"],
    #     },
    # )


# @configclass
# class RewardsCfg:
#     """Positive style reward, to scale metra reward."""

# instructor net
# instructor_guidance = RewTerm(func=mdp.instruction_guidance, params={"obs_name": "instructor"}, weight=1.0)

# # -- penalties
# lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
# ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
# dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
# dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
# action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
# feet_air_time = RewTerm(
#     func=mdp.feet_air_time,
#     weight=0.125,
#     params={
#         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
#         "command_name": "base_velocity",
#         "threshold": 0.5,
#     },
# )
# undesired_contacts = RewTerm(
#     func=mdp.undesired_contacts,
#     weight=-1.0,
#     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
# )

# # terminated = RewTerm(
# #     func=mdp.is_terminated_term,
# #     params={"term_keys": "upside_down"},
# #     weight=-1000.0,
# # )
# bad_orientation = RewTerm(
#     func=mdp.bad_orientation,
#     params={"limit_angle": math.radians(100)},
#     weight=-25.0,
# )

step_dt = 1 / 50


@configclass
class RewardsCfg:
    """No task reward, only style."""

    # is_alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=step_dt * -1.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=step_dt * -2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=step_dt * -5.0e-2)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=step_dt * 10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=step_dt * -30.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )

    undesired_contacts_shank = RewTerm(
        func=mdp.undesired_contacts,
        weight=step_dt * -30.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*SHANK"), "threshold": 1.0},
    )

    # undesired_contacts_base = RewTerm(
    #     func=mdp.is_terminated_term,
    #     params={"term_keys": "base_contact"},
    #     weight=-100.0,
    # )

    undesired_contacts_base = RewTerm(
        func=mdp.undesired_contacts,
        weight=step_dt * -30.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

    base_height = RewTerm(
        func=mdp.base_below_min_height,
        weight=-step_dt * 5.0,
        params={"target_height": 0.6},
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-step_dt * 10.0,
        params={"soft_ratio": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-step_dt * 2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    bad_orientation = RewTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(100)},
        weight=-step_dt * 10.0,
    )

    # terminated = RewTerm(
    #     func=mdp.is_terminated_term,
    #     params={"term_keys": "upside_down"},
    #     weight=-1000.0,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

    # too_far_away = DoneTerm(func=mdp.too_far_away, params={"max_dist": 15.0})

    # upside_down = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(100)})

    # goal_reached = DoneTerm(func=mdp.goal_reached, params={"threshold_dist": 0.5})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # remove_rewards = CurrTerm(
    #     func=mdp.anneal_reward_weight,
    #     params={
    #         "term_names": [
    #             "base_height",
    #             "joint_deviation",
    #             "bad_orientation",
    #         ],
    #         "ratio": 0.0,
    #         "start_step": 25_000,
    #         "num_steps": 70_000,
    #     },
    # )

    # anneal_rewards = CurrTerm(
    #     func=mdp.anneal_reward_weight,
    #     params={
    #         "term_names": [
    #             "undesired_contacts_thigh",
    #             "undesired_contacts_shank",
    #             "undesired_contacts_base",
    #         ],
    #         "ratio": 0.1,
    #         "start_step": 25_000,
    #         "num_steps": 70_000,
    #     },
    # )

    # num_obstacles = CurrTerm(func=mdp.num_boxes_curriculum)

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    # eye: tuple[float, float, float] = (0.0, -14.0, 6.0)
    eye: tuple[float, float, float] = (0.0, 3.0, 2.0)
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
class MetraAnymalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Data container
    data_container: mdp.DataContainer = mdp.DataContainer()

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=800.0)
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
        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 10.0  #
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # GPU settings
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        # self.sim.physx.gpu_collision_stack_size = 2**28
        # self.sim.physx.gpu_found_lost_pairs_capacity = 2**28

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.lidar is not None:
        #     self.scene.lidar.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
