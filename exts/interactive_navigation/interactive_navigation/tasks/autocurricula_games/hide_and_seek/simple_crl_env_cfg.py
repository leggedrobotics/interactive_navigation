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


##
# Task-specific configurations
##

from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.assets import (
    ROBOT_CFG,
    ROBOT_USD_CFG,
    CUBOID_CFG,
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
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=mdp.terrain.MESH_STEPPABLE_PYRAMID_TERRAIN_CFG,
        max_init_terrain_level=1000,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.0,
            dynamic_friction=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots

    robot: ArticulationCfg = ROBOT_USD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/yaw_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=10,
        ),
        max_distance=100.0,
        drift_range=(-0.0, 0.0),
        debug_vis=False,
        history_length=0,
        # mesh_prim_paths=["/World/ground", self.scene.obstacle.prim_path],
        mesh_prim_paths=[
            "/World/ground",
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Wall_.*", is_global=False),
        ],
        track_mesh_transforms=True,
        visualizer_cfg=SEGMENT_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

    height_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/yaw_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 30.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.5,
            size=(10, 8),
        ),
        max_distance=100.0,
        drift_range=(-0.0, 0.0),
        debug_vis=False,
        history_length=0,
        # mesh_prim_paths=["/World/ground", self.scene.obstacle.prim_path],
        mesh_prim_paths=[
            "/World/ground",
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Wall_.*", is_global=False),
        ],
        track_mesh_transforms=True,
        visualizer_cfg=SEGMENT_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

    lidar_top = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/yaw_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=10,
        ),
        max_distance=100.0,
        drift_range=(-0.0, 0.0),
        debug_vis=False,
        history_length=0,
        # mesh_prim_paths=["/World/ground", self.scene.obstacle.prim_path],
        mesh_prim_paths=[
            "/World/ground",
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Wall_.*", is_global=False),
        ],
        track_mesh_transforms=False,
        visualizer_cfg=SEGMENT_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCasterTop"),
    )

    # boxes_contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Box_.*", history_length=1, track_air_time=False)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
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
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # usd robot
    wrench = mdp.ArticulatedWrench2DActionCfg(asset_name="robot", debug_vis=True)

    jump = mdp.ArticulatedJumpActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # self
        # TODO: add robot height
        my_velocity = ObsTerm(
            func=mdp.velocity_2d_b,
            params={"entity_cfg": SceneEntityCfg("robot"), "pov_entity_cfg": SceneEntityCfg("robot")},
        )

        actions = ObsTerm(func=mdp.last_action)

        lidar_scan = ObsTerm(
            func=mdp.lidar_obs_dist_2d,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.0, 100.0),
        )

        lidar_scan_top = ObsTerm(
            func=mdp.lidar_obs_dist_2d,
            params={"sensor_cfg": SceneEntityCfg("lidar_top")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.0, 100.0),
        )

        pose_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class PolicyGoalCfg(ObsGroup):
        """Observations for policy goal group.
        This may be constant"""

        # TODO how to implement this for the constant case? Then we do not need to constantly update the  it,
        # and we also dont need a replay buffer for it

        pose_goal = ObsTerm(func=mdp.actor_goal.generated_goal, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(ObsGroup):
        """Observation group for the critic."""

        # self
        my_velocity = ObsTerm(
            func=mdp.velocity_2d_b,
            params={"entity_cfg": SceneEntityCfg("robot"), "pov_entity_cfg": SceneEntityCfg("robot")},
        )

        # actions = ObsTerm(func=mdp.last_action)

        lidar_scan = ObsTerm(
            func=mdp.lidar_obs_dist_2d,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.0, 100.0),
        )

        lidar_scan_top = ObsTerm(
            func=mdp.lidar_obs_dist_2d,
            params={"sensor_cfg": SceneEntityCfg("lidar_top")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.0, 100.0),
        )

        # privileged information
        heigh_scan = ObsTerm(
            func=mdp.lidar_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scan")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
        )

        pose_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticGoalCfg(ObsGroup):
        """Observations for policy goal group
        Note. here we may have observations that are also in the state. These here however are used when sampling a future state.
        Therefore its not trivial to predict this from the current state.
        I.e. : state contains goal. -> state to goal is trivial.
        state does not contain future goal. -> state to future goal is not trivial."""

        pose_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy_obs: PolicyCfg = PolicyCfg()
    policy_goal: PolicyGoalCfg = PolicyGoalCfg()

    critic_state: CriticStateCfg = CriticStateCfg()
    critic_goal: CriticGoalCfg = CriticGoalCfg()


Z_ROBOT = 0.3 + 0.05
Z_BOX = 0.25 + 0.05
Z_WALL = 0.5 + 0.05


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_all = EventTerm(func=mdp.reset_scene_to_default)

    reset_robot_random = EventTerm(
        func=mdp.reset_root_state_uniform_on_terrain_aware,
        mode="reset",
        params={
            "pose_range": {"yaw": (0, 0)},
            # "lowest_level": True,
            "offset": [0.0, 0.0, Z_ROBOT],
            "reset_used_patches_ids": True,
        },
    )

    # reset_robot_fix = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={"pose_range": {"x": (11.0, 11.0), "y": (11.0, 11.0), "z": (Z_ROBOT, Z_ROBOT)}, "velocity_range": {}},
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_yaw_joint = EventTerm(
        func=mdp.reset_id_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-3.1416, 3.1416),  # (-3.1416, 3.1416),
            "velocity_range": (0.0, 0.0),
            "joint_names": ["joint_yaw"],
        },
    )


@configclass
class RewardsCfg:
    """No rewards for CRL."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    num_obstacles = CurrTerm(func=mdp.num_boxes_curriculum)

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    # eye: tuple[float, float, float] = (-60.0, 0.5, 70.0)
    eye: tuple[float, float, float] = (9.7, 9.7, 8.0)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    # lookat: tuple[float, float, float] = (-60.0, 0.0, -10000.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: Literal["world", "env", "asset_root"] = "env"
    """
    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    """
    env_index: int = 0
    asset_name: str | None = None  # "robot"


##
# Environment configuration
##


@configclass
class CrlTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Data container
    data_container: mdp.DataContainer = mdp.DataContainer()

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
        # general settings
        self.decimation = 10  # 20 Hz
        self.episode_length_s = 45.0
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # GPU settings
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_collision_stack_size = 2**30

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
