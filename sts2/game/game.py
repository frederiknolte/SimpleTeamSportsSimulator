# Copyright (C) 2020 Electronic Arts Inc.  All rights reserved.

"""
This is a fairly simple python program that implements a simplified sports game.
It is intended to be used for reinforcment learning experiments that capture the
core problems of sports AI without all the messy details of the full game.  
The simulation should run thousands of times faster than the game so collecting 
reinforcement data and validating RL and IL methods will be much quicker.  
Once discovered we can apply them to the full title.
"""

import os
import numpy
import pandas as pd
import random
import datetime
import json

from sts2.game.simulation import Simulation, GameEvent, GameHistoryEntry
from sts2.game.arena import Arena
from sts2.game.control import Control
from sts2.game.game_state import GameState, Action
from sts2.game.physics import Physics
from sts2.game.rules import Rules, STANDARD_GAME_RULES
from sts2.game.settings import GamePhase, STS2Event, Outputs, TeamSide


class Game(Simulation):
    GOAL_REWARD = 1.0

    def __init__(self, players, rules=None, verbosity=0, save_states=False, client_adapter_cls=None):
        super(Game, self).__init__(players, verbosity)
        self.client_adapter = client_adapter_cls(self)
        self.save_states = save_states
        self.team_players = []
        self.team_players.append([x for x in players if x.team_side == TeamSide.HOME])
        self.team_players.append([x for x in players if x.team_side == TeamSide.AWAY])
        if rules is None:
            rules = STANDARD_GAME_RULES
        self.rules = rules
        self.arena = Arena(rules.arena_size)
        self.physics = Physics(self)

        self.state = GameState(self)
        self.control = Control(self)
        self.state.SetField(GameState.PREVIOUS_PHASE, GamePhase.PRE_GAME, init=True)
        self.state.SetField(GameState.CURRENT_PHASE, GamePhase.PRE_GAME, init=True)

        # More of the MAS additions
        self.players_by_distance_to_ball_by_team = {}
        self.init_exp = 1.0

    def CustomTick(self):
        vb = max(0, self.verbosity - 1)

        # from base class but we want it logged
        self.state.series = self.state.series.copy()
        self.state.series.tick = self.tick

        self.player_action_list = [None] * len(self.players)
        self.player_reward_list = [0.0] * len(self.players)
        self.player_policy_list = [None] * len(self.players)
        self.player_value_estimate_list = [0.0] * len(self.players)

        # if (self.tick + 1) % self.client_adapter.max_tick_per_episode == 0:
        #     print('FORCED RESET')
        #     self.SetGamePhase(GamePhase.STOPPAGE_GOAL)

        self.PhaseUpdate(vb)

        self.DrawArena(vb)

        self.BallUpdate(vb)
        self.AIUpdate(vb)
        self.LocomotionUpdate(vb)
        self.physics.Update(vb)
        self.ActionUpdate(vb)
        self.RulesUpdate(vb)
        self.CheckGoal(vb)
        self.physics.BoardBallCollisionUpdate(verbosity=vb)

        if vb:
            print('tick %3d %10s -> %10s' % (
                self.tick, self.GetPreviousGamePhase(), self.GetGamePhase()))

    def InitPlayerPositions(self):
        sampled_positions = []
        for player in self.players:
            while True:
                # r = numpy.random.random()
                r = numpy.random.uniform(0, 0.5)
                r = r ** self.init_exp
                attack_z = player.GetAttackingNetPos(self)[1]
                z = attack_z * r - attack_z * (1.0 - r)
                position = [float(numpy.random.randint(self.arena.min_x, self.arena.max_x)), z]
                if not self.physics.CollisionTest(position, sampled_positions, self.rules.player_radius):
                    break

            player.SetPosition(self, numpy.array(position))
            sampled_positions.append(position)

    def RandomlyGiveControl(self):
        # random player seemed to mostly pick the first player
        # now alternating teams and picking random player
        team = int((self.state.series.home_score + self.state.series.away_score)) % 2
        if len(self.team_players[team]) == 0:
            team = TeamSide.Opposite(team)
        target = random.choice(self.team_players[team])
        self.control.GiveControl(target)
        target.ResponseTime(self, self.rules.receive_response_time)

    def GetGamePhase(self):
        return self.state.GetField(GameState.CURRENT_PHASE)

    def SetGamePhase(self, new_phase, verbosity=0):
        old_phase = self.GetGamePhase()
        if verbosity: print('%s -> %s' % (old_phase, new_phase))
        self.state.SetField(GameState.PREVIOUS_PHASE, old_phase)
        self.state.SetField(GameState.CURRENT_PHASE, new_phase)

    def GetPreviousGamePhase(self):
        return self.state.GetField(GameState.PREVIOUS_PHASE)

    def OnPlayStart(self):
        for player in self.players:
            player.OnPlayStart(self)

        self.InitPlayerPositions()
        self.InitBall()
        self.RandomlyGiveControl()
        self.round_start_tick = self.tick

    def InitBall(self):
        self.state.SetBallMechanism(0)
        self.state.SetBallSendDirection(numpy.zeros(2))

    def GetScore(self, teamside):
        return self.state.GetTeamField(teamside, GameState.TEAM_SCORE)

    def SetScore(self, teamside, score):
        self.state.SetTeamField(teamside, GameState.TEAM_SCORE, score)

    def PhaseUpdate(self, verbosity):
        if self.GetGamePhase() == GamePhase.PRE_GAME:
            self.SetGamePhase(GamePhase.START_PLAY, verbosity)
            self.game_event_history.AddEvent(
                GameEvent(self.tick, STS2Event.GAME_START, '', ''))
            self.PhaseUpdate(verbosity)
        elif self.GetGamePhase() == GamePhase.START_PLAY:
            self.OnPlayStart()
            self.SetGamePhase(GamePhase.GAME_ON, verbosity)
            self.PhaseUpdate(verbosity)
        elif self.GetGamePhase() == GamePhase.GAME_ON:
            if self.tick >= self.rules.max_tick:
                self.SetGamePhase(GamePhase.STOPPAGE_TIMEUP, verbosity)
                self.game_event_history.AddEvent(
                    GameEvent(self.tick, STS2Event.GAME_END, '', ''))
                self.PhaseUpdate(verbosity)
            elif (self.tick - self.round_start_tick) >= self.rules.max_round_tick:
                # Current round has been going on for too long
                self.WipeCurrentGame()
                self.SetGamePhase(GamePhase.START_PLAY, verbosity)
                self.PhaseUpdate(verbosity)
            else:
                # this just clears out the "previous_phase" properly
                self.SetGamePhase(GamePhase.GAME_ON, verbosity)
        elif self.GetGamePhase() == GamePhase.STOPPAGE_GOAL:
            self.SetGamePhase(GamePhase.START_PLAY, verbosity)
            self.PhaseUpdate(verbosity)
        elif self.GetGamePhase() == GamePhase.STOPPAGE_TIMEUP:
            self.SetGamePhase(GamePhase.GAME_OVER, verbosity)
            self.PhaseUpdate(verbosity)
        elif self.GetGamePhase() == GamePhase.GAME_OVER:
            if self.save_states:
                self.SaveStateHistory()
        else:
            raise TypeError('unknown game phase', self.GetGamePhase())

    def InputToPolicyVectorIndex(self, attack_dir, input):
        # compensate so policy space is normalized to attack dir
        input *= attack_dir
        input = numpy.sign(input)
        try:
            pvi = int((input[0] + 1) * 3 + input[1] + 1)
        except:
            print(f'input: {input}')
            print(f'attack_dir: {attack_dir}')
            pvi = 0
        # print("input", input, "policy vector index", pvi)

        # input 0 0 is 4 (no input)
        # input 0 1 is 5 (up)
        # input -1 1 is 2 (up right)
        # input -1 0 is 1
        # input 1 -1 is 6 (down left)

        return pvi

    def PolicyVectorIndexToInput(self, attack_dir, ls_index):
        # should be the reverse of the above
        input = numpy.zeros(2)
        input[0] = ls_index // 3 - 1
        input[1] = ls_index % 3 - 1
        input *= attack_dir

        if self.verbosity: print("input", input[0], input[1], "ls_index", ls_index)
        return input

    def PlayerDecisionsToRLStates(self, player):
        action = player.GetAction(self)
        action_index = Action.ACTION_LIST.index(action)
        policy_vector = numpy.zeros(Outputs.NUM)  # TODO

        if action in Action.PASSES:
            policy_vector[Outputs.PASS] = 1.0
        elif action is Action.SHOOT:
            policy_vector[Outputs.SHOOT] = 1.0
        else:
            policy_vector[Outputs.SKATE] = 1.0

        # map discrete ls into the first 9 policy outputs
        i = self.InputToPolicyVectorIndex(player.GetAttackDir(self), player.GetInput(self))
        policy_vector[i] = 1.0

        value_estimate = 0.0  # TODO - would come from NN evaluation
        return action_index, policy_vector, value_estimate

    def sort_by_distance_to_ball(self):
        """ Sort all players by team and distance to the puck owner aka 'controller'. """
        self.players_by_distance_to_controller_by_team = {0: [], 1: []}
        puck_position = self.state.GetBallPosition()
        for player in self.players:
            dist = numpy.linalg.norm(puck_position - player.GetPosition(self))
            self.players_by_distance_to_controller_by_team[player.team_side].append(
                (dist, player.name))
        self.players_by_distance_to_controller_by_team[0] = {
            entry[1]: ii for ii, entry in
            enumerate(sorted(self.players_by_distance_to_controller_by_team[0]))
        }
        self.players_by_distance_to_controller_by_team[1] = {
            entry[1]: ii for ii, entry in
            enumerate(sorted(self.players_by_distance_to_controller_by_team[1]))
        }

    def AIUpdate(self, verbosity):
        self.sort_by_distance_to_ball()
        for i, player in zip(range(len(self.players)), self.players):
            player.Think(self, verbosity)
            self.player_action_list[i], self.player_policy_list[i], self.player_value_estimate_list[
                i] = self.PlayerDecisionsToRLStates(player)

    def LocomotionUpdate(self, verbosity):
        for player in self.players:
            control_input = player.GetInput(self)
            player.RunMotionModel(self, control_input, player.GetActionTime(self))
            if self.rules.layout_constraint is Rules.LayoutConstraint.CROSSOVER_CONSTRAINT:
                i = player.GetTeamIndex(self)
                x1 = 0.2
                x2 = 0.6
                y1 = 0.0
                y2 = 1.0

                # mirror for 2nd player
                if i % 2:
                    x1, x2 = 1.0 - x1, 1.0 - x2

                x, y = self.arena.GetNormalizedCoord(player.GetPosition(self))

                if player.team_side == TeamSide.AWAY:
                    x, y = 1.0 - x, 1.0 - y

                x_prime = x1 + (x2 - x1) * y / (y2 - y1)

                if player.team_side == TeamSide.AWAY:
                    x_prime, y = 1.0 - x_prime, 1.0 - y

                constrained_pos = self.arena.GetArenaCoordFromNormalized(numpy.array([x_prime, y]))
                player.SetPosition(self, constrained_pos)

    def GetCapableTeamPlayers(self, team):
        return [player for player in self.team_players[team] if player.GetActionTime(self) == 0]

    def BallUpdate(self, vb):
        if self.control.GetControl() is None:
            # Ball is traveling in the air
            # Update possession
            closest_player = None
            closest_dist = numpy.infty
            for player in self.players:
                if player.GetActionTime(self) <= 0:
                    diff = self.state.GetBallPosition() - player.GetPosition(self)
                    dist = numpy.linalg.norm(diff)
                    if dist <= self.rules.max_intercept_dist and dist < closest_dist:
                        closest_player = player
                        closest_dist = dist

            if closest_player is not None:
                # Ball is caught
                self.control.GiveControl(closest_player)
                self.state.SetBallMechanism(7)
                closest_player.ResponseTime(self, self.rules.receive_response_time)
                return

            # Update position
            position = self.state.GetBallPosition()
            if numpy.all(self.state.GetBallSendDirection() == 0):
                # Ball has been traveling for at least one time step
                velocity = self.state.GetBallVelocity() * self.rules.ball_velocity_decay
            else:
                # Ball has been released by controler in previous step but still contains old velocity information on how it got to current step
                velocity = self.state.GetBallSendDirection() * self.rules.ball_speed
                self.state.SetBallSendDirection(numpy.zeros(2))
            position += velocity
            self.state.SetBallPosition(position)
            self.state.SetBallVelocity(velocity)
            self.state.SetBallMechanism(8)

        else:
            # Ball is in control
            # Update position
            prev_control_pos = self.control.GetControl().GetPosition(self)
            prev_ball_pos = self.state.GetBallPosition()
            self.state.SetBallPosition(prev_control_pos)
            self.state.SetBallVelocity(prev_control_pos - prev_ball_pos)
            self.state.SetBallMechanism(7)

    def ComputeOnNetChance(self, player):
        net_delta = player.GetAttackingNetPos(self) - player.GetPosition(self)
        net_dir = net_delta / numpy.linalg.norm(net_delta)
        shot_directness_chance = numpy.abs(net_dir[1])
        shot_distance_chance = min(1.0, self.rules.shot_distance_accuracy_scale / numpy.abs(
            net_delta[1]))
        # print('shot', net_delta, net_dir, shot_directness_chance, shot_distance_chance, shot_directness_chance * shot_distance_chance)
        return shot_directness_chance * shot_distance_chance

    def PlayerShot(self, player, simulate, verbosity):
        if not simulate:
            assert (self.control.GetControl() is player)

        if self.control.GetControl() is None:
            return 0.0

        on_net_chance = self.ComputeOnNetChance(player)

        through_chance = self.physics.InterceptTest(player.GetPosition(self),
                                                    player.GetAttackingNetPos(self),
                                                    self.GetCapableTeamPlayers(
                                                        TeamSide.Opposite(
                                                            self.control.GetControl().team_side)),
                                                    max(0, verbosity - 1))

        if not simulate:
            self.game_event_history.AddEvent(
                GameEvent(self.tick, STS2Event.SHOT, player.name, ''))

            player.ResponseTime(self, self.rules.shot_response_time)
            self.SendBall(player.GetAttackingNetPos(self))

        return through_chance * on_net_chance

    def PlayerPass(self, source_player, target_player, simulate, verbosity):
        assert (self.control.GetControl() is source_player)
        through_chance = self.physics.InterceptTest(source_player.GetPosition(self),
                                                                 target_player.GetPosition(self),
                                                                 self.GetCapableTeamPlayers(
                                                                     TeamSide.Opposite(
                                                                         self.control.GetControl().team_side)),
                                                                 max(0, verbosity - 1))
        if not simulate:
            self.game_event_history.AddEvent(
                GameEvent(self.tick, STS2Event.PASS, source_player.name, target_player.name))
            self.SendBall(target_player.GetPosition(self))
            source_player.ResponseTime(self, self.rules.pass_response_time)
        return through_chance

    def CompleteCheck(self, control_player, checking_player):
        assert (self.control.GetControl() is control_player)
        self.game_event_history.AddEvent(
            GameEvent(self.tick, STS2Event.CHECK, checking_player.name, control_player.name))

        control_player.Stun(self, self.rules.check_stun_time)
        # self.control.GiveControl(checking_player)
        # checking_player.ResponseTime(self, self.rules.receive_response_time)
        self.control.RemoveControl()
        # TODO change ball behaviour after check (ball should go towards checking player, shouldn't it?)
        # TODO we could also argue that the ball just goes in the previous direction before the controller was tackled.
        # TODO this would not be relevant anyway most of the time as the checking player would usually be close enough to gain
        # TODO possession at the next time step

    def AwardGoal(self, team_side, verbose):
        self.game_event_history.AddEvent(
            GameEvent(self.tick, STS2Event.GOAL, TeamSide.GetName(team_side), ''))
        self.SetScore(team_side, self.GetScore(team_side) + 1)
        self.SetGamePhase(GamePhase.STOPPAGE_GOAL)
        for i, p in zip(range(len(self.players)), self.players):
            if p.team_side == team_side:
                self.player_reward_list[i] = self.GOAL_REWARD
            else:
                self.player_reward_list[i] = -self.GOAL_REWARD
        if verbose: print(f'Goal by team {TeamSide.GetName(team_side)}.')

    def CheckGoal(self, verbose):
        ball_pos = self.state.GetBallPosition()
        arena_side = numpy.sign(ball_pos[1])
        ball_pos[1] += arena_side * self.rules.ball_radius
        prev_ball_pos = ball_pos - self.state.GetBallVelocity()

        for team_side in TeamSide.TEAMSIDES:
            team_net_x = self.state.GetTeamField(team_side, self.state.TEAM_NET_X)
            team_net_z = self.state.GetTeamField(team_side, self.state.TEAM_NET_Z)
            left_post = numpy.array([team_net_x - 0.5, team_net_z])
            right_post = numpy.array([team_net_x + 0.5, team_net_z])
            goal = self.physics.IntersectionTest(prev_ball_pos, ball_pos, left_post, right_post)

            if goal:
                self.AwardGoal(team_side, verbose)
                self.control.Reset(self)

    def ActionUpdate(self, verbosity):
        control_player = self.control.GetControl()
        if control_player:
            if control_player.GetAction(self) is Action.SHOOT:
                self.PlayerShot(control_player, False, max(0, verbosity - 1))
            else:
                teammates = self.team_players[control_player.team_side]
                for i, teammate, pass_action in zip(range(len(teammates)), teammates,
                                                    Action.PASSES):
                    if teammate is control_player:
                        continue
                    if control_player.GetAction(self) is pass_action:
                        self.PlayerPass(control_player, teammate, False, max(0, verbosity - 1))

    def SendBall(self, target_position):
        self.control.RemoveControl()
        diff = target_position - self.state.GetBallPosition()
        length = numpy.linalg.norm(diff)
        direction = diff / length
        self.state.SetBallSendDirection(direction)

    def RulesUpdate(self, verbosity):
        pass

    def IsSimulationComplete(self):
        # should return true if simulation is complete
        return False  # self.GetGamePhase() is GamePhase.GAME_OVER

    def ShowState(self):
        pass

    def GetHashableGameStateVector(self):
        return self.state.series

    def SaveStateHistory(self):
        date = datetime.date.today().isoformat()
        os.makedirs(os.path.join('.', 'datasets', date), exist_ok=True)
        save_state_path = os.path.join('.', 'datasets', date, 'STATEHISTORY.json')
        state_history = [state.state.to_dict() for state in self.game_state_history]

        with open(save_state_path, 'w') as fout:
            json.dump(state_history, fout)

    def LoadStateHistory(self, load_path, start=0):
        with open(load_path, 'r') as fin:
            state_history = json.load(fin)

        for history_entry in state_history[start:]:
            h = GameHistoryEntry(tick=None,
                                 state=pd.Series(history_entry),
                                 player_identity_list=None,
                                 player_policy_list=None,
                                 player_action_list=None,
                                 player_value_estimate_list=None,
                                 player_reward_list=None)
            self.game_state_history.append(h)

    def DrawArena(self, vb):
        if not vb:
            return

        def ArenaCoordToArrayCoord(arena, p):
            return p[1] - arena.min_z + 1, p[0] - arena.min_x + 1

        def AddCharToArenaString(arena, a, ch, p):
            coord = ArenaCoordToArrayCoord(arena, p)
            a[int(round(coord[0])), int(round(coord[1]))] = ch

        # since arrays are (row, col) but rows are in Z we need to reverse these
        a = numpy.ndarray((self.arena.arena_size[1] + 2, self.arena.arena_size[0] + 2), dtype='<U1')
        a.fill(' ')
        a[0, :] = '-'
        a[-1, :] = '-'
        a[:, 0] = '|'
        a[:, -1] = '|'
        a[0, 0] = '/'
        a[-1, -1] = '/'
        a[0, -1] = '\\'
        a[-1, 0] = '\\'

        for gp in self.arena.net_position:
            c = ArenaCoordToArrayCoord(self.arena, gp)
            a[c[0], c[1]] = 'G'

        cl = ['x', 'o']

        for player_list, ch in zip(self.team_players, cl):
            for player, num in zip(player_list, range(1, len(player_list) + 1)):
                nums = str(num)
                if self.control.HasControl(player):
                    ch2 = ch.upper()
                    # nums = ' '
                elif player.GetAction(self) == Action.STUNNED and player.GetActionTime(
                        self) > 0:
                    ch2 = '_'
                else:
                    ch2 = ch
                for x in range(int(-self.rules.player_radius), int(self.rules.player_radius) + 1):
                    for z in range(int(-self.rules.player_radius),
                                   int(self.rules.player_radius) + 1):
                        position = player.GetPosition(self)
                        x2, z2 = (x + position[0], z + position[1])
                        if numpy.linalg.norm(
                                numpy.array([x2, z2]) - position) <= self.rules.player_radius:
                            AddCharToArenaString(self.arena, a, ch2, (x2, z2))
                        AddCharToArenaString(self.arena, a, nums, (position[0], position[1]))

        if self.control.GetControl() is None:
            position = self.state.GetBallPosition()
            AddCharToArenaString(self.arena, a, 'B', (position[0], position[1]))

        s = ''
        for row in a:
            s = ' '.join(row) + '\n' + s

        if self.verbosity:
            print(s)
            print('tick %d of %d home score:%d away score:%d' % (
                self.tick, self.rules.max_tick, self.GetScore(TeamSide.HOME),
                self.GetScore(TeamSide.AWAY)))
