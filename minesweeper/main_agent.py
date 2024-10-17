import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication

from const import ROOT
from minesweeper.game.gui_agent import MinesweeperGameWAgent
from solver.dqn_agent import DQNAgent
from solver.dqn_agent_fed import FedDQNAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', required=True, choices=['normal2', 'fed'], help='Agent type')
    parser.add_argument('-r', '--rows', default=6, type=int, help='Number of rows in the Minesweeper grid')
    parser.add_argument('-c', '--cols', default=6, type=int, help='Number of columns in the Minesweeper grid')
    parser.add_argument('-m', '--mines', default=5, type=int, help='Number of mines in the Minesweeper grid')
    parser.add_argument('--ckpt', default=os.path.join(ROOT, "checkpoint"), help='Checkpoint path')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screen = max(app.screens(), key=lambda screen: screen.size().width() * screen.size().height())
    logical_dpi = screen.logicalDotsPerInch()
    if args.agent == 'normal2':
        window = MinesweeperGameWAgent(args.rows, args.cols, args.mines, logical_dpi=logical_dpi, agent_cls=DQNAgent, checkpoint_dir=args.ckpt)
    elif args.agent == 'fed':
        window = MinesweeperGameWAgent(args.rows, args.cols, args.mines, logical_dpi=logical_dpi, agent_cls=FedDQNAgent, checkpoint_dir=args.ckpt)
    else:
        raise ValueError(f'Invalid agent type: {args.agent}')
    window.show()
    sys.exit(app.exec_())
