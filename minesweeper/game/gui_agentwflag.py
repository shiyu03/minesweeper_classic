import os

from const import ROOT
from game.gui_agent import MinesweeperGameWAgent
from solver.dqn_agent_wflag import DQNAgentWFlag


class MinesweeperGameWAgentWFlag(MinesweeperGameWAgent):
    def __init__(self, rows=16, cols=16, mines=40, *, logical_dpi):
        super().__init__(rows, cols, mines, logical_dpi=logical_dpi)

    def initSolver(self, rows, cols, mines):
        input_shape = (rows, cols)
        output_size = rows * cols * 2
        self.agent = DQNAgentWFlag(input_shape, output_size)
        checkpoint_dir = os.path.join(ROOT, "checkpoint_wflag")
        try:
            self.agent.load(checkpoint_dir, rows, cols, mines)
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        self.agent.model.eval()

    def solverMove(self):
        valid_actions_reveal, valid_actions_flag = self.env.get_valid_actions_wflag()
        state = self.env.get_normalized_state_wflag()
        if self.env.first_click:
            action = self.agent.act(state, valid_actions_reveal, force_random=True)
        else:
            action = self.agent.act(state, valid_actions_reveal + valid_actions_flag)
        row, col, flag = action
        print(f"Agent move: {row}, {col}, {flag=}")
        self.makeMoveHndlr(row, col, flag=flag==1, show_last_action=True, allow_click_revealed_num=False, allow_recursive=False)()
