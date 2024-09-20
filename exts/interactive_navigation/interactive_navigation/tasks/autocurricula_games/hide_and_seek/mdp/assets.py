import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, AssetBase

# - robot:
ROBOT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.SphereCfg(
        radius=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    collision_group=-1,
)

# - movable objects:

# - cuboid:
CUBOID_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cuboid",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    collision_group=-1,
)

# - wall:
WALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Wall",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 2.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    collision_group=-1,
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
