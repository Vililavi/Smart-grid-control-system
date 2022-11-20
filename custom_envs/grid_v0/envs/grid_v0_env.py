"""
Microgrid-simulation to OpenAI gym
V 0.1
"""
import os
import math
from random import randint
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.registration import EnvSpec
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

from microgrid_sim.environment import get_default_microgrid_env


class GridV0Env(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    Gym environment for the microgrid.

    """

    spec = EnvSpec(
        id='Grid-v0',
        entry_point='grid_v0.envs:GridV0Env',
        max_episode_steps=24,
    )

    def __init__(self, max_total_steps: int):
        self._max_total_steps = max_total_steps

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self._data_path = os.path.join(project_dir, "data")
        start_idx = randint(0, 14600 - self._max_total_steps)
        self._env = get_default_microgrid_env(self._data_path, start_idx)
        self.state = None
        self._step = 0

        low = np.array(
            [
                0.0,    # TCL SoC
                0.0,    # ESS SoC
                -22.0,  # out temperature
                0.0,    # generated energy
                0.0,    # up price
                0.0,    # base residential load
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                1.0,     # TCL SoC
                1.0,     # ESS SoC
                32.0,    # out temperature
                1800.0,  # generated energy
                2999.0,  # up price
                1.4,     # base residential load (max 1.4 by default)
            ]
            ,
            dtype=np.float32,
        )

        max_steps = self.spec.max_episode_steps
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(low, high, dtype=np.float32),                   # float values (listed above)
                spaces.Discrete(2 * max_steps + 5, start=-2 * max_steps),  # pricing counter
                spaces.Discrete(24),                                       # hour of day
            ]
        )
        self.action_space = spaces.MultiDiscrete([4, 5, 2, 2])

    def step(self, action: spaces.MultiDiscrete):
        """
        Takes action as an input, returns new state and reward. The action consists of:
            TCL action:               [0:3]
            Price action:             [0:4]
            Energy deficiency action: [0:1]
            Energy excess action:     [0:1]

        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        action_tuple = (action[0], action[1] - 2, action[2], action[3])
        self.state, reward = self._env.step(action_tuple)

        self._step += 1
        terminated = self._step >= self.spec.max_episode_steps
        if self._step > self.spec.max_episode_steps:
            logger.warn("Called 'step()' on terminated environment!")

        return (np.array(self.state[:6], dtype=np.float32), self.state[6], self.state[7]), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Resets the environment to a starting state.
        """
        super().reset(seed=seed)

        start_idx = randint(0, 14600 - self._max_total_steps)
        self._env = get_default_microgrid_env(self._data_path, start_idx)
        self.state = None
        self._step = 0

        self.state = self._env.get_state()

        return (np.array(self.state[:6], dtype=np.float32), self.state[6], self.state[7]), {}

    def render(self):
        """
        For rendering a visualization, not used.
        """

    def close(self):
        """
        For closing visualization, not used.
        """
