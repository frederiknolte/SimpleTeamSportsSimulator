# Copyright (C) 2020 Electronic Arts Inc.  All rights reserved.
import numpy

from sts2.game.game_state import GameState
from sts2.game.settings import STS2Event
from sts2.game.simulation import GameEvent


class Control:
    def __init__(self, game):
        self.game = game

    def Reset(self, game):
        self.game = game

    def GiveControl(self, player):
        index = self.game.team_players[player.team_side].index(player)
        self.game.state.SetField(GameState.CONTROL_INDEX, index)
        self.game.state.SetField(GameState.CONTROL_TEAM, player.team_side)
        control_pos = self.game.state.GetPlayerPosition(player)
        prev_ball_pos = self.game.state.GetBallPosition()
        self.game.state.SetBallPosition(control_pos)
        self.game.state.SetBallVelocity(control_pos - prev_ball_pos)
        self.game.state.SetBallSendDirection(numpy.zeros(2))
        self.game.game_event_history.AddEvent(
            GameEvent(self.game.tick, STS2Event.GAIN_CONTROL, player.name, ''))

    def RemoveControl(self):
        self.game.state.SetField(GameState.CONTROL_TEAM, -1)
        self.game.state.SetField(GameState.CONTROL_INDEX, -1)

    def GetControl(self):
        control_team = int(self.game.state.GetField(GameState.CONTROL_TEAM))
        control_player = int(self.game.state.GetField(GameState.CONTROL_INDEX))
        if control_player != -1 and control_team != -1:
            return self.game.team_players[control_team][control_player]
        else:
            return None

    def HasControl(self, player):
        return player is self.GetControl()
