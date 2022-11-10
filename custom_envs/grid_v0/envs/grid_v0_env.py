"""
Microgrid-simulaatio OpenAI gymiin
V 0.1
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class GridV0Env(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    Katso esimerkkiä --> custom_cartpole_env.py

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        """
        Alustus environmentille. Tähän tilaan palataan resetillä.
        """


    def step(self, action):
        """
        Tämä on se varsinainen työrukkanen, jossa itse simuloinnin ajaminen tapahtuu.

        Step ottaa inputtina neljä subactionia, joista jokainen voi saada tietyt arvot (ohjeistuksen mukaan):
            TCL action:             [0:3]
            Price action:           [0:4]
            Energy defiency action: [0:1]
            Energy excess action:   [0:1]

        Näistä toki vielä tarkistettava että

        Palauttaa staten ja rewardin
        """

        terminated = True
        reward = 1.0

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Palauttaa aloitustilan. Jonkin verran tätäkin tarvinnee muuutella, mut ei ehkä niin paljoa.
        """
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        """
        Renderöintimetodi. Voidaan jättää tekeminen viimeiseksi tai kokonaan pois, ei pakollinen.
        """

    def close(self):
        """
        En ole ihan varma tarvitseeko tätä lähinnä renderöinnin lopettamiseen. Ehkä.
        """