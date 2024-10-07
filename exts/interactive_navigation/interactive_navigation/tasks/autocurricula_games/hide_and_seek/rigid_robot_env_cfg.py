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


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=mdp.terrain.MESH_PYRAMID_TERRAIN_CFG,
        max_init_terrain_level=2,
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
        debug_vis=True,
    )
    # robots
    robot: RigidObjectCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # assets:
    box_1: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_1", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_2: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_2", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_3: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_3", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_4: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_4", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_5: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_5", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_6: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_6", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_7: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_7", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    box_8: RigidObjectCfg = CUBOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Box_8", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )

    # sensors
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
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
            RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Wall_.*", is_global=False),
        ],
        track_mesh_transforms=True,
        visualizer_cfg=SEGMENT_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

    lidar_top = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
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
            RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Box_.*", is_global=False),
            # RayCasterCfg.RaycastTargetCfg(target_prim_expr="/World/envs/env_.*/Wall_.*", is_global=False),
        ],
        track_mesh_transforms=True,
        visualizer_cfg=SEGMENT_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCasterTop"),
    )

    boxes_contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Box_.*", history_length=1, track_air_time=False)

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

    # base_velocity = mdp.UniformPose2dCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     simple_heading=True,
    #     debug_vis=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-20, 20), pos_y=(-0.5, 0.5), heading=(-0.0, 0.0)),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    # base_agent = mdp.SimpleActionCfg(
    #     asset_name="robot",
    #     debug_vis=True,
    # )

    # rigid body robot
    simple_wrench = mdp.WrenchAction2DCfg(
        asset_name="robot",
        debug_vis=True,
    )

    simple_jump = mdp.JumpActionCfg(
        asset_name="robot",
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # self
        my_velocity = ObsTerm(
            func=mdp.velocity_2d_b,
            params={"entity_cfg": SceneEntityCfg("robot"), "pov_entity_cfg": SceneEntityCfg("robot")},
        )

        # lidar_scan = ObsTerm(
        #     func=mdp.lidar_obs_dist,
        #     params={"sensor_cfg": SceneEntityCfg("lidar")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(0.0, 100.0),
        # )

        # lidar_scan_top = ObsTerm(
        #     func=mdp.lidar_obs_dist,
        #     params={"sensor_cfg": SceneEntityCfg("lidar_top")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(0.0, 100.0),
        # )

        # boxes:
        boxes_poses = ObsTerm(
            func=mdp.box_pose,
            params={
                "entity_str": "box",
                "pov_entity": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


Z_ROBOT = 0.3 + 0.05
Z_BOX = 0.25 + 0.05
Z_WALL = 0.5 + 0.05


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default)

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform_on_terrain_aware,
        mode="reset",
        params={
            "pose_range": {"yaw": (-3.14, 3.14)},
            "lowest_level": True,
            "offset": [0.0, 0.0, Z_ROBOT],
            "reset_used_patches_ids": True,
        },
    )

    reset_boxes = EventTerm(
        func=mdp.reset_root_state_uniform_on_terrain_aware,
        mode="reset",
        params={
            "pose_range": {
                "yaw": (-3.14, 3.14),
            },
            "lowest_level": True,
            "offset": [0.0, 0.0, Z_BOX],
            # "asset_cfg": SceneEntityCfg("box_1"),
            "asset_configs": [
                SceneEntityCfg("box_1"),
                SceneEntityCfg("box_2"),
                SceneEntityCfg("box_3"),
                SceneEntityCfg("box_4"),
                SceneEntityCfg("box_5"),
                SceneEntityCfg("box_6"),
                SceneEntityCfg("box_7"),
                SceneEntityCfg("box_8"),
            ],
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # rewards
    close_to_box = RewTerm(
        func=mdp.CloseToBoxReward().close_to_box_reward,
        weight=1,
        params={"threshold": 1.0},
    )

    jump = RewTerm(
        func=mdp.JumpReward().successful_jump_reward,
        weight=1000,
        params={},
    )

    high_up = RewTerm(
        func=mdp.high_up,
        weight=0.01,
        params={"height_range": (0.0, 3.0)},
    )

    # penalty
    outside = RewTerm(
        func=mdp.outside_env,
        weight=-1,
        params={"threshold": 12.5 * 2**0.5},
    )

    action_penalty = RewTerm(
        func=mdp.action_penalty,
        weight=-0.01,
        params={},
    )


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

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    # eye: tuple[float, float, float] = (-60.0, 0.5, 70.0)
    eye: tuple[float, float, float] = (0.0, 0.0, 20.0)
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
class RigidRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Data container
    data_container: mdp.DataContainer = mdp.DataContainer()
    data_container.num_obstacles_range = (3, 6)

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
        self.decimation = 10  # 10 Hz
        self.episode_length_s = 30.0
        # simulation settings
        # self.sim.dt = 0.005  # 200 Hz
        self.sim.dt = 0.005  # 100 Hz
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
