# Copyright (C) 2020 Electronic Arts Inc.  All rights reserved.

import random
import time
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
                          num_home_AdaptedSimplePlayer=0,
                          num_away_AdaptedSimplePlayer=0,
                          num_home_EgoisticPlayer=0,
                          num_away_EgoisticPlayer=0,
                          num_home_AggressivePlayer=0,
                          num_away_AggressivePlayer=0,
                          num_home_DefensivePlayer=0,
                          num_away_DefensivePlayer=0,
                          num_home_ShyPlayer=0,
                          num_away_ShyPlayer=0,
                          with_pygame=True,
                          save_states=False,
                          timeout_ticks=1e10,
                          verbosity=0)
    env.seed(42)
    obs, info = env.reset()

    start_time = time.time()
    while True:
        action = naive_action(obs)
        obs, r, done, info = env.step(None)
        env.render()

        if done: break

    print(f'Runtime: {time.time() - start_time} seconds.')
