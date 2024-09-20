# from interactive_navigation.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# from omni.isaac.lab.utils import configclass

# ##
# # Pre-defined configs
# ##
# from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip


# @configclass
# class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # switch robot to anymal-d
#         self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
