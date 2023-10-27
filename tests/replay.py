import json

from sts2.game.game import Game
from sts2.client_adapter import ClientAdapter
from sts2.game.pygame_interface import PygameInterface, INTERFACE_SETTINGS

state_history_path = 'datasets/2simple_v_2simple_1_old/train.json'

game = Game([], client_adapter_cls=ClientAdapter)
with open(state_history_path, 'r') as fin:
    game.LoadStateHistory(json.load(fin))

game_interface = PygameInterface(game, save_states=False, settings=INTERFACE_SETTINGS, replay=True)

while True:
    game_interface.update()
    game_interface.HandleGameReplayFrame()


