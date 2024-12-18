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
N_BOXES = 4


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=mdp.terrain.MESH_PYRAMID_TERRAIN_CFG,
        max_init_terrain_level=1,
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
        debug_vis=True,
    )
    # robots

    robot: ArticulationCfg = ROBOT_USD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/yaw_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 30.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.5,
            size=(5, 5),
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
            # add boxes with lidar sensors (only used for reward computation)
            setattr(self, f"box_{i}", CUBOID_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/Box_{i}"))
            setattr(
                self,
                f"box_lidar_bot_{i}",
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Box_{i}",
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
            setattr(
                self,
                f"box_lidar_top_{i}",
                getattr(self, f"box_lidar_bot_{i}").replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0))),
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

    # usd robot
    wrench = mdp.ArticulatedWrench2DActionCfg(
        asset_name="robot",
        debug_vis=True,
        max_velocity=2.5,
        max_vel_sideways=1.0,
        max_rotvel=2.0,
    )

    jump = mdp.ArticulatedJumpActionCfg(asset_name="robot")


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

        goal_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        height_scan = ObsTerm(
            func=mdp.lidar_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scan")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-10.0, 10.0),
        )

        # boxes:
        boxes_poses = ObsTerm(
            func=mdp.box_pose,
            params={
                "entity_str": "box",
                "pov_entity": SceneEntityCfg("robot"),
                "return_mask": True,
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

    reset_box_n_robot = EventTerm(
        func=mdp.reset_box_near_step_and_robot_near_box,
        mode="reset",
        params={
            "pose_range": {"yaw": (0, 0)},
            "box_asset_cfg": SceneEntityCfg("box_1"),
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "random_dist": True,
        },
    )

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
            "position_range": (-3.1416, 3.1416),
            "velocity_range": (0.0, 0.0),
            "joint_names": ["joint_yaw"],
        },
    )

    def __post_init__(self):
        for i in range(2, N_BOXES + 1):
            # add reset box events, each box one level higher than the previous one
            setattr(
                self,
                f"reset_box_{i}_near_step",
                EventTerm(
                    func=mdp.reset_near_step,
                    mode="reset",
                    params={
                        "pose_range": {"yaw": (0, 0)},
                        "asset_cfg": SceneEntityCfg(f"box_{i}"),
                        "level": i - 1,
                        "random_dist": True,
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
            "dist_sensor_1_str": "box_lidar_bot",
            "dist_sensor_2_str": "box_lidar_top",
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
            "dist_sensor_1_str": "box_lidar_bot",
            "dist_sensor_2_str": "box_lidar_top",
            "proximity_threshold": 0.5,
            "proximity_std": 1.0,
            "step_size_threshold": 0.75,
        },
    )

    close_to_box = RewTerm(
        func=mdp.CloseToBoxReward().close_to_box_reward,
        weight=0.1,
        params={"threshold": 1.0},
    )

    # Jumping
    successful_jump = RewTerm(
        func=mdp.JumpReward().successful_jump_reward,
        weight=10,
        params={},
    )

    # Moving up
    new_height = RewTerm(
        func=mdp.JumpReward().new_height_reached_reward,
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
    action_penalty = RewTerm(
        func=mdp.action_penalty,
        weight=-0.01,
        params={},
    )

    too_far = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
        params={"term_keys": "too_far_from_goal"},
        weight=-250.0,
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    wasting_time = RewTerm(
        func=mdp.is_alive,
        weight=-0.005,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out_perturbed, params={"perturbation": 5}, time_out=True)  # type: ignore

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


DIST_CURR = mdp.DistanceCurriculum(
    min_box_step_dist=0.2,
    min_robot_box_dist=2.0,
    max_box_step_dist=5.0,
    max_robot_box_dist=15.0,
    box_step_dist_increment=0.1,
    robot_box_dist_increment=0.1,
)

TERRAIN_CURR = mdp.TerrainCurriculum(
    num_successes=10, num_failures=10, goal_termination_name="goal_reached", random_move_prob=0.05
)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # num_obstacles = CurrTerm(func=mdp.num_boxes_curriculum)

    box_from_step_dist_curriculum = CurrTerm(func=DIST_CURR.box_from_step_dist_curriculum)

    robot_from_box_dist_curriculum = CurrTerm(func=DIST_CURR.robot_from_box_dist_curriculum)

    # robot_speed = CurrTerm(
    #     func=mdp.robot_speed_curriculum,
    #     params={"action_term_name": "wrench", "num_steps": 25_000, "start_multiplier": 2.0},
    # )

    terrain_levels = CurrTerm(func=TERRAIN_CURR.terrain_levels)


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
class MoveUpBoxesEnvCfg(ManagerBasedRLEnvCfg):
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

        # general settings
        self.decimation = 10  # 10 Hz
        self.episode_length_s = 30.0
        # simulation settings
        # self.sim.dt = 0.005  # 200 Hz
        self.sim.dt = 0.01  # 100 Hz
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
        if self.scene.height_scan is not None:
            self.scene.height_scan.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
