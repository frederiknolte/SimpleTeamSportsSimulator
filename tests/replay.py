import json

from sts2.game.game import Game
from sts2.client_adapter import ClientAdapter
from sts2.game.pygame_interface import PygameInterface, INTERFACE_SETTINGS

# state_history_path = 'datasets/2023-09-01/STATEHISTORY_processed.json'
state_history_path = '../../sparse-worldmodel_anson_new/checkpoints/r7zws8o3/predicted_states.json'

game = Game([], client_adapter_cls=ClientAdapter)
with open(state_history_path, 'r') as fin:
    game.LoadStateHistory(json.load(fin))

game_interface = PygameInterface(game, save_states=False, settings=INTERFACE_SETTINGS, replay=True)

while True:
    game_interface.update()
    game_interface.HandleGameReplayFrame()


