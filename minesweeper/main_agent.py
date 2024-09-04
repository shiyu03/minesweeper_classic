import sys
from PyQt5.QtWidgets import QApplication

from minesweeper.game.gui_agent import MinesweeperGameWAgent

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # TODO scale
    screen = max(app.screens(), key=lambda screen: screen.size().width() * screen.size().height())
    logical_dpi = screen.logicalDotsPerInch()
    resolution = screen.size()
    window = MinesweeperGameWAgent(6, 6, 5, logical_dpi=logical_dpi)
    window.show()
    sys.exit(app.exec_())