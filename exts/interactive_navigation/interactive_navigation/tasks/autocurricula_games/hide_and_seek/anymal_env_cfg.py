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
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )

    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        goal_radius_range=(2.5, 15.0),
        resampling_time_range=(1e9, 1e9),
        only_heading=False,
        only_position=True,
        attitude=True,
        debug_vis=True,
        env_frame=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        my_pose = ObsTerm(
            func=mdp.pose_3d_env,  # velocity_2d_b, rotation_velocity_2d_b
            params={"entity_cfg": SceneEntityCfg("robot")},
        )

        my_velocity = ObsTerm(
            func=mdp.velocity_3d_w,  # velocity_2d_b, rotation_velocity_2d_b
            params={"entity_cfg": SceneEntityCfg("robot")},
        )

        # actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # dist_to_goal = ObsTerm(
        #     func=mdp.dist_to_goal, params={"entity_cfg": SceneEntityCfg("robot"), "command_name": "robot_goal"}
        # )

        # goal_pos = ObsTerm(func=mdp.actor_goal.generated_goal, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PolicyGoalCfg(ObsGroup):
        """Goal to condition the policy.
        This may be constant. This will not be recorded in the replay buffer.
        Therefore it should be in the dedicated policy goal group and not in the policy obs group."""

        # TODO how to implement this for the constant case? Then we do not need to constantly update the  it,
        # and we also dont need a replay buffer for it

        goal_pos = ObsTerm(func=mdp.actor_goal.generated_goal, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(ObsGroup):
        """Observations for the critic."""

        # self
        my_pose = ObsTerm(
            func=mdp.pose_3d_env,  # velocity_2d_b, rotation_velocity_2d_b
            params={"entity_cfg": SceneEntityCfg("robot")},
        )
        my_velocity = ObsTerm(
            func=mdp.velocity_3d_w,  # velocity_2d_b, rotation_velocity_2d_b
            params={"entity_cfg": SceneEntityCfg("robot")},
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # actions = ObsTerm(func=mdp.last_action)
        # the critic gets the action anyway... TODO: check if this really is not needed
        # dist_to_goal = ObsTerm(
        #     func=mdp.dist_to_goal, params={"entity_cfg": SceneEntityCfg("robot"), "command_name": "robot_goal"}
        # )
        # goal_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CriticGoalCfg(ObsGroup):
        """Observations for policy goal group
        Note. here we may have observations that are also in the state. These here however are used when sampling a future state.
        Therefore its not trivial to predict (an embedding of) this from the current state.
        I.e. : - state contains goal. -> state to goal is trivial.
               - state does not contain future goal. -> state to future goal is not trivial."""

        goal_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_goal"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    # policy: PolicyCfg = PolicyCfg()
    policy_obs: PolicyCfg = PolicyCfg()
    policy_goal: PolicyGoalCfg = PolicyGoalCfg()

    critic_state: CriticStateCfg = CriticStateCfg()
    critic_goal: CriticGoalCfg = CriticGoalCfg()


reset_value = 0.0
reset_value_pos = 0.01


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_all = EventTerm(func=mdp.reset_scene_to_default)

    # reset_robot = EventTerm(
    #     func=mdp.reset_random_dist_from_goal,
    #     mode="reset",
    #     params={
    #         "radius_range": (0.0, 0.01),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "command_name": "robot_goal",
    #         "z_offset": 1.0,
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-reset_value_pos, reset_value_pos),
                "y": (-reset_value_pos, reset_value_pos),
                "yaw": (-0.0, 0.0),
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

    # reset_robot_yaw_joint = EventTerm(
    #     func=mdp.reset_id_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-3.1416, 3.1416),  # (-3.1416, 3.1416),
    #         "velocity_range": (0.0, 0.0),
    #         "joint_names": ["joint_yaw"],
    #     },
    # )


@configclass
class RewardsCfg:
    """No rewards for CRL."""

    # close_to_goal = RewTerm(func=mdp.close_to_goal, params={"command_name": "robot_goal"}, weight=1.0)

    # at_goal = RewTerm(func=mdp.at_goal, params={"command_name": "robot_goal"}, weight=100.0)

    # move_to_goal = RewTerm(func=mdp.moving_towards_goal, params={"command_name": "robot_goal"}, weight=0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # too_far_away = DoneTerm(func=mdp.too_far_away, params={"max_dist": 15.0})

    # upside_down = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.5})

    # goal_reached = DoneTerm(func=mdp.goal_reached, params={"threshold_dist": 0.5})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    num_obstacles = CurrTerm(func=mdp.num_boxes_curriculum)

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    # eye: tuple[float, float, float] = (-60.0, 0.5, 70.0)
    eye: tuple[float, float, float] = (20.0, 50.0, 25.0)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    # lookat: tuple[float, float, float] = (-60.0, 0.0, -10000.0)
    lookat: tuple[float, float, float] = (-25.0, 50.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: Literal["world", "env", "asset_root"] = "world"
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
class CrlAnymalEnvCfg(ManagerBasedRLEnvCfg):
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
        self.decimation = 1  # 20 Hz
        self.episode_length_s = 50.0
        # simulation settings
        self.sim.dt = 0.05  # 20 Hz
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
