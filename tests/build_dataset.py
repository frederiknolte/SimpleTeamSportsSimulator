import os
import json
import h5py
import numpy as np

from sts2.game.settings import TeamSide
from sts2.game.game_state import Action


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, "w") as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                if type(array_dict[i][key]) is dict:
                    for key2 in array_dict[i][key].keys():
                        grp.create_dataset(
                            "info_" + key2, data=array_dict[i][key][key2]
                        )
                else:
                    grp.create_dataset(key, data=array_dict[i][key])


def preprocess_ball_control_indicator(history):
    # When losing control of the ball, the team indicator is kept in the ball state for one additional time step.
    # This way, the ball knows which team to look at to evaluate a possible pass.
    history[0]['ball_control_id'] = history[0]['control_team']
    for i in range(1, len(history)):
        if history[i - 1]['control_team'] != -1 and history[i]['control_team'] == -1 \
                and history[i - 1]['control_index'] != -1 and history[i]['control_index'] == -1:
            history[i]['ball_control_id'] = history[i - 1]['ball_control_id']
        else:
            history[i]['ball_control_id'] = history[i]['control_team']
    return history


def gather_player_information(state, prefix):
    adapted_state = list()
    state_info = list()

    num_players = int(state[prefix + '_players'])
    for player_id in range(num_players):
        id = [0, int(prefix == 'home'), int(prefix == 'away')]  # ball/home/away flag
        team_possession = [0, 0]
        control_flag = [int(state['control_index'] == player_id and state['control_team'] == TeamSide.GetID(prefix))]
        action_time = [state[prefix + str(player_id) + '_action_time']]
        position = [state[prefix + str(player_id) + '_pos_x'], state[prefix + str(player_id) + '_pos_z']]
        velocity = [state[prefix + str(player_id) + '_vel_x'], state[prefix + str(player_id) + '_vel_z']]
        acceleration = [state[prefix + str(player_id) + '_input_x'], state[prefix + str(player_id) + '_input_z']]
        adapted_state.append(id + team_possession + control_flag + action_time + position + velocity + acceleration)

        mechanism = [state[prefix + str(player_id) + '_mechanism']]
        action = [Action.ACTION_LIST.index(state[prefix + str(player_id) + '_action'])]
        state_info.append(mechanism + action)
    return adapted_state, state_info


if __name__ == "__main__":

    PATH = 'datasets/2023-09-07/STATEHISTORY.json'
    NEW_PATH = os.path.join(*PATH.split('/')[:-1], 'dataset.h5')

    with open(PATH, 'r') as fin:
        json_file = json.load(fin)

    json_file = preprocess_ball_control_indicator(json_file)

    # Separate games
    state_history = [[]]
    for history_event in json_file:
        state_history[-1].append(history_event)
        if history_event['current_phase'] == 'STOPPAGE_GOAL':
            state_history.append(list())
    print(f'deleting last episode of length {len(state_history[-1])}')
    state_history.pop(-1)  # Remove unfinished/empty episode

    adapted_state_history = list()
    history_info = list()
    for episode in state_history:
        adapted_episode = list()
        episode_info = list()
        for state in episode:
            adapted_state = list()
            state_info = list()

            # Ball
            id = [1, 0, 0]  # ball/home/away flag
            team_possession = [int(state['ball_control_id'] == 0), int(state['ball_control_id'] == 1)]
            control_flag = [0]
            action_time = [0]
            position = [state['ball_pos_x'], state['ball_pos_z']]
            velocity = [state['ball_vel_x'], state['ball_vel_z']]
            acceleration = [0, 0]
            adapted_state.append(id + team_possession + control_flag + action_time + position + velocity + acceleration)

            mechanism = [state['ball_mechanism']]
            action = [Action.ACTION_LIST.index('NONE')]
            state_info.append(mechanism + action)

            # Players
            player_states, player_infos = gather_player_information(state, 'home')
            adapted_state.extend(player_states)
            state_info.extend(player_infos)
            player_states, player_infos = gather_player_information(state, 'away')
            adapted_state.extend(player_states)
            state_info.extend(player_infos)

            adapted_episode.append(adapted_state)
            episode_info.append(state_info)
        adapted_state_history.append(adapted_episode)
        history_info.append(episode_info)


    dataset = []
    for episode, info in zip(adapted_state_history, history_info):
        sample = dict()

        # State
        numpy_episode = np.array(episode)
        num_steps, num_obj = numpy_episode.shape[:2]
        sample['action'] = np.zeros(num_steps - 1).astype(np.int64)
        sample['obs'] = numpy_episode[:-1]
        sample['next_obs'] = numpy_episode[1:]

        # Additional info
        numpy_info = np.array(info)
        sample['info'] = {}
        sample['info'].update({f'{i}_mechanism': numpy_info[1:, i, 0] for i in range(num_obj)})
        sample['info'].update({f'{i}_i+1_action': numpy_info[1:, i, 1] for i in range(num_obj)})
        sample['info'].update({f'{i}_i_action': numpy_info[:-1, i, 1] for i in range(num_obj)})

        dataset.append(sample)

    save_list_dict_h5py(dataset, NEW_PATH)
