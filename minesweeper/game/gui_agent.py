import os

from const import ROOT
from game.gui_w_solver import MinesweeperGameWSolver
from solver.dqn_agent import DQNAgent


class MinesweeperGameWAgent(MinesweeperGameWSolver):
    def __init__(self, rows=16, cols=16, mines=40, *, logical_dpi):
        super().__init__(rows, cols, mines, logical_dpi=logical_dpi)

    def initSolver(self, rows, cols, mines):
        input_shape = (rows, cols)
        output_size = rows * cols
        self.agent = DQNAgent(input_shape, output_size)
        checkpoint_dir = os.path.join(ROOT, "checkpoint")
        try:
            self.agent.load(checkpoint_dir, rows, cols, mines)
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        self.agent.model.eval()

    def newAgent(self, rows, cols, mines):
        self.initSolver(rows, cols, mines)

    def solverMove(self):
        valid_actions = self.env.get_valid_actions()
        state = self.env.get_normalized_state()
        action = self.agent.act(state, valid_actions)
        row, col = action
        print(f"Agent move: {row}, {col}")
        self.makeMoveHndlr(row, col, show_last_action=True, allow_click_revealed_num=True, allow_recursive=False)()

    def newGame(self, rows, cols, mines):
        self.env.new_game(rows, cols, mines)
        self.newAgent(rows, cols, mines)
        self.updateMineLabel()
        self.initGame()
