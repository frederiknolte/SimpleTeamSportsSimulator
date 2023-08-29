# Copyright (C) 2020 Electronic Arts Inc.  All rights reserved.

import random
from sts2.environment import STS2Environment


def naive_action(obs):
    action = {}
    for k, v in obs.items():
        if isinstance(v, str) and "_ai_" in v:
            action[v] = {'action': 'NONE', 'input': [random.uniform(-1, 1), random.uniform(-1, 1)]}
    return action


if __name__ == "__main__":
    env = STS2Environment(num_home_SimplePlayer=3,
                          num_away_SimplePlayer=3,
                          with_pygame=True,
                          verbosity=0)
    obs, info = env.reset()
    while True:
        action = naive_action(obs)
        obs, r, done, info = env.step(None)
        env.render()

        if done: break
