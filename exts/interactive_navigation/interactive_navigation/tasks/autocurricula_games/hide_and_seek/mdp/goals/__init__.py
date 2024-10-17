""" 
Actor goals are used to condition the actor's policy on the goal. They can be constant (see: a single goal is all you need) 
or change due to a goal distribution or due to the robot moving in the environment. 
ie:  actor_goal = goal_func(goal_state)

Critic goals are used to train the critic. They are representations of future states, thus need to be updated at each step.
ie: critic_goal = goal_func(current_state)

IMPORTANT: botch actor and critic goals have to represent the exact same thing (identical goal_func), since the actor's goal is used during
rollouts, but the critic goals is not only used to train the critic, but also to generate the actor's goal when training the actor.

"""

from . import actor_goal
from . import critic_goal

__all__ = ["actor_goal", "critic_goal"]
