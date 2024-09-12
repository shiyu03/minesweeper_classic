from abc import abstractmethod

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton

from game.gui import MinesweeperGame


class MinesweeperGameWSolver(MinesweeperGame):
    def __init__(self, rows=16, cols=16, mines=40, *, logical_dpi):
        super().__init__(rows, cols, mines, logical_dpi=logical_dpi)
        self.initSolver(rows, cols, mines)
        self.initKeyPressListener()

    def initUI(self):
        super().initUI()
        self.solverButton = QPushButton('✨')
        self.solverButton.setFont(self.restartButton.font())
        self.solverButton.setFixedSize(self.restartButton.size())
        self.solverButton.clicked.connect(self.solverMove)
        # self.topLayout.removeWidget(self.empty)
        self.topLayout.addWidget(self.solverButton, 0, 2)

    @abstractmethod
    def initSolver(self, rows, cols, mines): ...

    @abstractmethod
    def solverMove(self): ...

    def makeMoveHndlr(self, row, col, flag: bool, show_last_action=True, allow_click_revealed_num=True, allow_recursive=True):
        return super().makeMoveHndlr(row, col, flag, show_last_action, allow_click_revealed_num, allow_recursive)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.solverMove()
        else:
            super().keyPressEvent(event)