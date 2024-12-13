import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, AssetBase
from omni.isaac.lab.actuators import ImplicitActuatorCfg

# - robot:
ROBOT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.CuboidCfg(
        size=(0.6, 0.6, 0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0, disable_gravity=False, max_linear_velocity=10.0, max_angular_velocity=90.0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.3),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)

ROBOT_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rafael/Projects/MT/interactive_navigation/exts/interactive_navigation/interactive_navigation/tasks/autocurricula_games/hide_and_seek/mdp/urdfs/output_2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=30.0,
            max_angular_velocity=30.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35), joint_pos={"joint_x": 0.0, "joint_y": 0.0, "joint_z": 0.0, "joint_yaw": 0.0}
    ),
    actuators={
        "x_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_x"],
            effort_limit=1000.0,
            velocity_limit=25.0,
            stiffness=0.0,
            damping=100.0,
            friction=0.5,
        ),
        "y_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_y"],
            effort_limit=10000.0,
            velocity_limit=25.0,
            stiffness=0.0,
            damping=100.0,
            friction=0.5,
        ),
        "z_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_z"],
            effort_limit=1000.0,
            velocity_limit=25.0,
            stiffness=0.0,
            damping=1.0,
            friction=1.0,
        ),
        "yaw_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_yaw"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=100.0,
            friction=0.5,
        ),
    },
)


# - movable objects:

# - cuboids:
CUBOID_FLAT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cuboid",
    spawn=sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.75),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, friction_combine_mode="average"
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)


CUBOID_SMALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cuboid",
    spawn=sim_utils.CuboidCfg(
        size=(0.25, 0.25, 0.25),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, friction_combine_mode="average"
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.3)),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)

CUBOID_BIG_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cuboid",
    spawn=sim_utils.CuboidCfg(
        size=(1.5, 1.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, friction_combine_mode="average"
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)

CUBOID_TALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cuboid",
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 0.3, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, friction_combine_mode="average"
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.9)),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)


# - wall:
WALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Wall",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 8.0, 2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0, disable_gravity=False, kinematic_enabled=True
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)

# TODO: Add more objects here, ramp


# markers:
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg

SEGMENT_RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)
