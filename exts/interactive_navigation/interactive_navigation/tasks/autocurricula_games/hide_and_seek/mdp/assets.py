import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, AssetBase

# - robot:
ROBOT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.CuboidCfg(
        size=(1.2, 0.8, 0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0, disable_gravity=False, max_linear_velocity=3.0, max_angular_velocity=90.0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)

# - movable objects:

# - cuboid:
CUBOID_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cuboid",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0, disable_gravity=False, max_angular_velocity=3.14, kinematic_enabled=True
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collision_group=0,
)

# - wall:
WALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Wall",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 3.5, 1.0),
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

MY_RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "ground_hits": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "obstacle_hits": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
