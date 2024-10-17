import numpy as np
import torch
import torch.nn.functional as F

from game.gui_w_solver import MinesweeperGameWSolver
from game.minesweeper_env import CellState
from solver.dqn_agent import DQNAgent


class MinesweeperGameWAgent(MinesweeperGameWSolver):
    def __init__(self, rows=16, cols=16, mines=40, *, logical_dpi, agent_cls, checkpoint_dir):
        self.agent_cls = agent_cls
        self.checkpoint_dir = checkpoint_dir
        super().__init__(rows, cols, mines, logical_dpi=logical_dpi)

    def initSolver(self, rows, cols, mines):
        self.agent = self.agent_cls(eval=True)
        try:
            try:
                self.agent.load_best(self.checkpoint_dir)
            except FileNotFoundError:
                self.agent.load(self.checkpoint_dir)
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        self.agent.model.eval()

    def newAgent(self, rows, cols, mines):
        self.initSolver(rows, cols, mines)

    def makeMoveHndlr(self, row, col, flag: bool, show_last_action=True, allow_click_revealed_num=True, allow_recursive=True):
        handler = super().makeMoveHndlr(row, col, flag, show_last_action, allow_click_revealed_num, allow_recursive)
        def _handler():
            handler()
            next_valid_actions = self.env.get_valid_actions()
            next_state = self.env.get_normalized_state()
            _, masked_q_values = self.agent.act(next_state, next_valid_actions)
            self.plotQValues(masked_q_values)
        return _handler

    def solverMove(self):
        valid_actions = self.env.get_valid_actions()
        state = self.env.get_normalized_state()
        if self.env.first_click:
            action, _ = self.agent.act(state, valid_actions, force_random=True)
        else:
            action, _ = self.agent.act(state, valid_actions)
        row, col = action
        print(f"Agent move: {row}, {col}")
        self.makeMoveHndlr(row, col, flag=False, show_last_action=True, allow_click_revealed_num=True, allow_recursive=False)()

    def newGame(self, rows, cols, mines):
        self.newAgent(rows, cols, mines)
        super().newGame(rows, cols, mines)

    def plotQValues(self, masked_q_values):
        if masked_q_values is not None:
            valid_q_values = masked_q_values[~masked_q_values.mask]

            # Split into positive and negative parts
            positive_q_values = valid_q_values[valid_q_values >= 0]
            negative_q_values = valid_q_values[valid_q_values < 0]

            # Apply softmax to each part
            positive_probs = F.softmax(torch.tensor(positive_q_values, dtype=torch.float32), dim=0).numpy()
            negative_probs = F.softmax(torch.tensor(-negative_q_values, dtype=torch.float32), dim=0).numpy()

            # Combine the results
            probabilities = np.zeros_like(valid_q_values, dtype=np.float32)
            probabilities[valid_q_values >= 0] = positive_probs
            probabilities[valid_q_values < 0] = -negative_probs

            # Map probabilities back to the original masked array
            probs_array = np.zeros_like(masked_q_values, dtype=np.float32)
            probs_array[~masked_q_values.mask] = probabilities
            probs_array[masked_q_values.mask] = 0

            probs_array = probs_array.reshape(self.env.rows, self.env.cols)
            for r in range(0, self.env.rows):
                for c in range(0, self.env.cols):
                    if not masked_q_values.mask[r, c]:
                        prob = probs_array[r, c]
                        if prob >= 0:  # Color green
                            rgb_tuple = (0, int(prob * 255), 0)
                        else:  # Color red
                            rgb_tuple = (int(-prob * 255), 0, 0)
                        radius = int(self.cells[r, c].width() / 2 * np.abs(prob))
                        self.cells[r, c].plotQValue(rgb_tuple, radius)

