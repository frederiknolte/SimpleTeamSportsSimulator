import os
import json
import numpy as np
np.random.seed(42)


PATH = 'datasets/2simple_v_2simple/train.json'
NUM_SAMPLES = 9
NEW_PATH = os.path.join(PATH.split('/')[0], PATH.split('/')[1] + f'_{NUM_SAMPLES}', PATH.split('/')[2])

with open(PATH, 'r') as fin:
    json_file = json.load(fin)


# Separate games
state_history = [[]]
for history_event in json_file:
    state_history[-1].append(history_event)
    if history_event['current_phase'] == 'STOPPAGE_GOAL':
        state_history.append(list())
print(f'deleting last episode of length {len(state_history[-1])}')
state_history.pop(-1)  # Remove unfinished/empty episode

subsamples_id = np.random.randint(low=0, high=len(state_history), size=NUM_SAMPLES)
subsamples = [state_history[i] for i in subsamples_id]

state_history = []
for sample in subsamples:
    state_history.extend(sample)

with open(NEW_PATH, 'w') as fout:
    json.dump(state_history, fout)
