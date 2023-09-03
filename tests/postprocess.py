import json
import numpy as np

from sts2.game.settings import TeamSide
from sts2.game.rules import DATACOLLECTION_GAME_RULES

RULES = DATACOLLECTION_GAME_RULES
state_history_path = 'datasets/2023-09-01/STATEHISTORY.json'
state_history_save_path = 'datasets/2023-09-01/STATEHISTORY_processed.json'

with open(state_history_path, 'r') as fin:
    combined_state_history = json.load(fin)

# Separate games
state_history = [[]]
for history_event in combined_state_history:
    state_history[-1].append(history_event)
    if history_event['current_phase'] == 'STOPPAGE_GOAL':
        state_history.append(list())
state_history.pop(-1)  # Remove unfinished/empty episode

# Set ball location in case of ball possession
for episode in state_history:
    for i in range(len(episode)):
        history_event = episode[i]
        control_prefix = TeamSide.GetName(history_event['control_team']) + str(history_event['control_index'])
        history_event['ball_pos_x'] = history_event[control_prefix + '_pos_x']
        history_event['ball_pos_z'] = history_event[control_prefix + '_pos_z']
        history_event['ball_in_air'] = False

# Adapt ball location in case of pass
for episode in state_history:
    for i in range(1, len(episode)):
        prev_history_event = episode[i-1]
        history_event = episode[i]
        passer_prefix = TeamSide.GetName(prev_history_event['control_team']) + str(prev_history_event['control_index'])

        if prev_history_event['control_team'] == -1 and prev_history_event['control_index'] == -1:
            continue

        if history_event[passer_prefix + '_action'].startswith('PASS'):
            receiver_index = int(history_event[passer_prefix + '_action'][5:]) - 1
            receiver_prefix = TeamSide.GetName(prev_history_event['control_team']) + str(receiver_index)
            pass_successful = (prev_history_event['control_team'] == history_event['control_team'] and
                               history_event['control_index'] == receiver_index)

            if not pass_successful:
                interceptor_prefix = TeamSide.GetName(history_event['control_team']) + \
                                     str(history_event['control_index'])

            # Compute ball trajectory
            passer_pos_x = history_event[passer_prefix + '_pos_x']
            passer_pos_z = history_event[passer_prefix + '_pos_z']
            receiver_pos_x = history_event[receiver_prefix + '_pos_x']
            receiver_pos_z = history_event[receiver_prefix + '_pos_z']
            diff = np.array([receiver_pos_x - passer_pos_x, receiver_pos_z - passer_pos_z])
            episode[i]['control_team'] = -1
            episode[i]['control_index'] = -1

            for j in range(1, RULES.airtime):
                offset = diff * j / RULES.airtime
                episode[i + j]['ball_pos_x'] = passer_pos_x + offset[0]
                episode[i + j]['ball_pos_z'] = passer_pos_z + offset[1]
                episode[i + j]['ball_in_air'] = True

                # Check whether ball is intercepted at current position
                if not pass_successful:
                    interceptor_ball_diff = np.array([episode[i + j]['ball_pos_x'] - episode[i + j][interceptor_prefix + '_pos_x'],
                                                      episode[i + j]['ball_pos_z'] - episode[i + j][interceptor_prefix + '_pos_z']])
                    interceptor_ball_dist = np.linalg.norm(interceptor_ball_diff)

                    # Ball is intercepted at current position
                    if interceptor_ball_dist < RULES.max_intercept_dist:
                        episode[i + j]['ball_pos_x'] = episode[i + j][interceptor_prefix + '_pos_x']
                        episode[i + j]['ball_pos_z'] = episode[i + j][interceptor_prefix + '_pos_z']
                        episode[i + j]['ball_in_air'] = False
                        break

                episode[i + j]['control_team'] = -1
                episode[i + j]['control_index'] = -1

# Add ball velocity
for episode in state_history:
    for i in range(len(episode) - 1):
        if episode[i]['ball_in_air'] and not episode[i + 1]['ball_in_air']:
            episode[i]['ball_vel_x'] = 0.0
            episode[i]['ball_vel_z'] = 0.0
        else:
            episode[i]['ball_vel_x'] = episode[i + 1]['ball_pos_x'] - episode[i]['ball_pos_x']
            episode[i]['ball_vel_z'] = episode[i + 1]['ball_pos_z'] - episode[i]['ball_pos_z']
episode[-1]['ball_vel_x'] = episode[-2]['ball_vel_x']
episode[-1]['ball_vel_z'] = episode[-2]['ball_vel_z']

# Flatten game episodes
flat_state_history = []
for episode in state_history:
    flat_state_history.extend(episode)

with open(state_history_save_path, 'w') as fout:
    json.dump(flat_state_history, fout)


# Set ball velocity
# do not save paused frames

# Handle interception
# TODO Handle actively taking the ball while ball in flight
# TODO handle shoot

# remove unfinished games
# ball being passed indicator
pass
