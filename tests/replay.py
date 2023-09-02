from sts2.game.game import Game
from sts2.client_adapter import ClientAdapter
from sts2.game.pygame_interface import PygameInterface, INTERFACE_SETTINGS

state_history_path = 'datasets/2023-09-01/STATEHISTORY_processed.json'

game = Game([], client_adapter_cls=ClientAdapter)
game.LoadStateHistory(state_history_path)

game_interface = PygameInterface(game, save_states=True, settings=INTERFACE_SETTINGS, replay=True)

while True:
    game_interface.update()
    game_interface.HandleGameReplayFrame()


