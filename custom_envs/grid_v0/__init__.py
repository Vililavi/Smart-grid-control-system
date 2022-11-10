from gym.envs.registration import register

register(id='Grid-v0', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='grid_v0.envs:GridV0Env' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)