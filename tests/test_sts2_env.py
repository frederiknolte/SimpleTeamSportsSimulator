# Copyright (C) 2020 Electronic Arts Inc.  All rights reserved.

import random
import tqdm
from sts2.environment import STS2Environment


SEED = 42
WITH_PYGAME = False
SAVE_STATES = True
TIMEOUT_TICKS = 50000


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
                          with_pygame=WITH_PYGAME,
                          save_states=SAVE_STATES,
                          timeout_ticks=TIMEOUT_TICKS,
                          verbosity=0)
    env.seed(SEED)
    obs, info = env.reset()
    progress_bar = tqdm.tqdm(total=TIMEOUT_TICKS)

    while True:
        action = naive_action(obs)
        obs, r, done, info = env.step(None)
        env.render()
        progress_bar.update(n=env.game.tick - progress_bar.n)
        if done: break
