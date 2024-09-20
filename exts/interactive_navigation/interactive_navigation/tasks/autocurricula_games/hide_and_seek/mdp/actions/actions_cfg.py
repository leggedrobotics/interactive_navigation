from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .actions import SimpleAction


@configclass
class SimpleActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SimpleAction
