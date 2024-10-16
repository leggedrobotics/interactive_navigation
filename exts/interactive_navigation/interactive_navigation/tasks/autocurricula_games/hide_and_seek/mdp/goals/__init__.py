""" 
Actor goals are used to condition the actor's policy on the goal. They can be constant (see: a single goal is all you need)
Critic goals are used to train the critic. They are representations of future states, thus need to be updated at each step.

IMPORTANT: botch actor and critic goals have to represent the exact same thing, since the actor's goal is used during
rollouts, but the critic goals is not only used to train the critic, but also to generate the actor's goal when training the actor.

"""

from . import actor_goal
from . import critic_goal

__all__ = ["actor_goal", "critic_goal"]
