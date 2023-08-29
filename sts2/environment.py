# Copyright (C) 2020 Electronic Arts Inc.  All rights reserved.

import random
import numpy as np

from sts2.client_adapter import ClientAdapter
from sts2.game.game import Game
from sts2.game.game_state import Action
from sts2.game.player import SimplePlayer, AdaptedSimplePlayer, EgoisticPlayer, AggressivePlayer, DefensivePlayer, ShyPlayer
from sts2.game.pygame_interface import PygameInterface, INTERFACE_SETTINGS
from sts2.game.rules import STANDARD_GAME_RULES
from sts2.game.settings import TeamSide


class AgentPlayer(SimplePlayer):
    def __init__(self, name, team_side):
        super().__init__(name, team_side)

    def custom_think(self, game, verbosity):
        discrete_action, continuous_input = game.client_adapter.unpack_action(self)
        if discrete_action is None:
            discrete_action = Action.NONE
        else:
            discrete_action = getattr(Action, discrete_action)
        self.SetAction(game, discrete_action)
        self.SetInput(game, continuous_input)


def get_game(timeout_ticks,
             num_home_agents,
             num_away_agents,
             num_home_SimplePlayer,
             num_away_SimplePlayer,
             num_home_AdaptedSimplePlayer,
             num_away_AdaptedSimplePlayer,
             num_home_EgoisticPlayer,
             num_away_EgoisticPlayer,
             num_home_AggressivePlayer,
             num_away_AggressivePlayer,
             num_home_DefensivePlayer,
             num_away_DefensivePlayer,
             num_home_ShyPlayer,
             num_away_ShyPlayer,
             verbosity=0):
    # Prepare players
    i = 0
    home_players = []
    for _ in range(num_home_agents):
        i += 1
        home_players.append(AgentPlayer('h_ai_' + str(i), TeamSide.HOME))
    for _ in range(num_home_SimplePlayer):
        i += 1
        home_players.append(SimplePlayer('h_sim_' + str(i), TeamSide.HOME))
    for _ in range(num_home_AdaptedSimplePlayer):
        i += 1
        home_players.append(AdaptedSimplePlayer('h_adsim_' + str(i), TeamSide.HOME))
    for _ in range(num_home_EgoisticPlayer):
        i += 1
        home_players.append(EgoisticPlayer('h_ego_' + str(i), TeamSide.HOME))
    for _ in range(num_home_AggressivePlayer):
        i += 1
        home_players.append(AggressivePlayer('h_agg_' + str(i), TeamSide.HOME))
    for _ in range(num_home_DefensivePlayer):
        i += 1
        home_players.append(DefensivePlayer('h_def_' + str(i), TeamSide.HOME))
    for _ in range(num_home_ShyPlayer):
        i += 1
        home_players.append(ShyPlayer('h_shy_' + str(i), TeamSide.HOME))

    i = 0
    away_players = []
    for _ in range(num_away_agents):
        i += 1
        away_players.append(AgentPlayer('a_ai_' + str(i), TeamSide.AWAY))
    for _ in range(num_away_SimplePlayer):
        i += 1
        away_players.append(SimplePlayer('a_sim_' + str(i), TeamSide.AWAY))
    for _ in range(num_away_AdaptedSimplePlayer):
        i += 1
        away_players.append(AdaptedSimplePlayer('a_adsim_' + str(i), TeamSide.AWAY))
    for _ in range(num_away_EgoisticPlayer):
        i += 1
        away_players.append(EgoisticPlayer('a_ego_' + str(i), TeamSide.AWAY))
    for _ in range(num_away_AggressivePlayer):
        i += 1
        away_players.append(AggressivePlayer('a_agg_' + str(i), TeamSide.AWAY))
    for _ in range(num_away_DefensivePlayer):
        i += 1
        away_players.append(DefensivePlayer('a_def_' + str(i), TeamSide.AWAY))
    for _ in range(num_away_ShyPlayer):
        i += 1
        away_players.append(ShyPlayer('a_shy_' + str(i), TeamSide.AWAY))

    # Rules
    rules = STANDARD_GAME_RULES
    rules.max_tick = int(timeout_ticks)

    return Game(home_players + away_players, rules, verbosity=verbosity,
                client_adapter_cls=ClientAdapter)


def get_pygame(game):
    return PygameInterface(game, INTERFACE_SETTINGS)


class STS2Environment(object):

    def __init__(
            self,
            *,
            num_home_agents=0,
            num_away_agents=0,
            num_home_SimplePlayer=3,
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
            with_pygame=False,
            timeout_ticks=1e10,
            verbosity=0):

        self.game = get_game(
            timeout_ticks=timeout_ticks,
            num_home_agents=num_home_agents,
            num_away_agents=num_away_agents,
            num_home_SimplePlayer=num_home_SimplePlayer,
            num_away_SimplePlayer=num_away_SimplePlayer,
            num_home_AdaptedSimplePlayer=num_home_AdaptedSimplePlayer,
            num_away_AdaptedSimplePlayer=num_away_AdaptedSimplePlayer,
            num_home_EgoisticPlayer=num_home_EgoisticPlayer,
            num_away_EgoisticPlayer=num_away_EgoisticPlayer,
            num_home_AggressivePlayer=num_home_AggressivePlayer,
            num_away_AggressivePlayer=num_away_AggressivePlayer,
            num_home_DefensivePlayer=num_home_DefensivePlayer,
            num_away_DefensivePlayer=num_away_DefensivePlayer,
            num_home_ShyPlayer=num_home_ShyPlayer,
            num_away_ShyPlayer=num_away_ShyPlayer,
            verbosity=verbosity)

        self.pygame = get_pygame(self.game) if with_pygame else None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        observation = self.game.client_adapter.send_state()
        return observation, ''

    def render(self):
        if self.pygame:
            self.pygame.HandleGameReplayFrame()

    def update(self):
        if self.pygame:
            self.pygame.update()
        else:
            self.game.update()

    def step(self, action):
        reward = None
        info = None

        self.game.client_adapter.receive_action(action)

        self.update()

        observation = self.game.client_adapter.send_state()
        done = observation.get('current_phase') == "GAME_OVER"
        return observation, reward, done, info
