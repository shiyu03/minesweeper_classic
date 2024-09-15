import argparse
import os
import sys
from PyQt5.QtWidgets import QApplication

from const import ROOT
from minesweeper.game.gui_agent import MinesweeperGameWAgent, MinesweeperGameWAgentRecur
from minesweeper.game.gui_agentwflag import MinesweeperGameWAgentWFlag, MinesweeperGameWAgentWFlagRecur

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', required=True, choices=['normal', 'normalrecur', 'wflag', 'wflagrecur'], help='Agent type')
    parser.add_argument('-r', '--rows', default=6, type=int, help='Number of rows in the Minesweeper grid')
    parser.add_argument('-c', '--cols', default=6, type=int, help='Number of columns in the Minesweeper grid')
    parser.add_argument('-m', '--mines', default=5, type=int, help='Number of mines in the Minesweeper grid')
    parser.add_argument('--ckpt', default=os.path.join(ROOT, "checkpoint"), help='Checkpoint path')
    args = parser.parse_args()
    args.ckpt = os.path.join(args.ckpt, f'dqn_{args.rows}x{args.cols}x{args.mines}')

    app = QApplication(sys.argv)
    screen = max(app.screens(), key=lambda screen: screen.size().width() * screen.size().height())
    logical_dpi = screen.logicalDotsPerInch()
    if args.agent == 'normal':
        window = MinesweeperGameWAgent(args.rows, args.cols, args.mines, logical_dpi=logical_dpi, checkpoint_dir=args.ckpt)
    elif args.agent == 'normalrecur':
        window = MinesweeperGameWAgentRecur(args.rows, args.cols, args.mines, logical_dpi=logical_dpi, checkpoint_dir=args.ckpt)
    elif args.agent == 'wflag':
        window = MinesweeperGameWAgentWFlag(args.rows, args.cols, args.mines, logical_dpi=logical_dpi, checkpoint_dir=args.ckpt)
    elif args.agent == 'wflagrecur':
        window = MinesweeperGameWAgentWFlagRecur(args.rows, args.cols, args.mines, logical_dpi=logical_dpi, checkpoint_dir=args.ckpt)
    else:
        raise ValueError(f'Invalid agent type: {args.agent}')
    window.show()
    sys.exit(app.exec_())
