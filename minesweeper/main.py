import sys
from PyQt5.QtWidgets import QApplication

from minesweeper.game.gui import MinesweeperGame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # TODO scale
    screen = max(app.screens(), key=lambda screen: screen.size().width() * screen.size().height())
    logical_dpi = screen.logicalDotsPerInch()
    resolution = screen.size()
    # print(logical_dpi, resolution) # office: 96, 1920x1080
    window = MinesweeperGame(6, 6, 5, logical_dpi=logical_dpi)
    window.show()
    sys.exit(app.exec_())