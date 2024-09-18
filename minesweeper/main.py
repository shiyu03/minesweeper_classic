import argparse
import sys
from PyQt5.QtWidgets import QApplication

from minesweeper.game.gui import MinesweeperGame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rows', default=6, type=int, help='Number of rows in the Minesweeper grid')
    parser.add_argument('-c', '--cols', default=6, type=int, help='Number of columns in the Minesweeper grid')
    parser.add_argument('-m', '--mines', default=5, type=int, help='Number of mines in the Minesweeper grid')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    # TODO scale
    screen = max(app.screens(), key=lambda screen: screen.size().width() * screen.size().height())
    logical_dpi = screen.logicalDotsPerInch()
    resolution = screen.size()
    # print(logical_dpi, resolution) # office: 96, 1920x1080
    window = MinesweeperGame(args.rows, args.cols, args.mines, logical_dpi=logical_dpi)
    window.show()
    sys.exit(app.exec_())