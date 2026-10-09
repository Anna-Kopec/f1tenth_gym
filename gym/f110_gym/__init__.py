"""!
@package f110_gym
@brief F1TENTH gym: a lightweight 2D racing simulator with the OpenAI gym interface.

Importing this package registers the environment with gym under the id
@c 'f110-v0', so it can be created with:
@code
env = gym.make('f110_gym:f110-v0', map='vegas', num_agents=1)
@endcode
The @c 'f110_gym:' prefix makes gym import this package (and so run the
registration below) before looking up the id.

@see f110_env.py: F110Env for the keyword arguments and the step / reset interface.
"""
from gym.envs.registration import register
register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
	)
