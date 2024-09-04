import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QVBoxLayout

from game.gui import Cell, CellState


class TestMineButton(QMainWindow):
    def __init__(self):
        super().__init__()
        self.buttons: dict[tuple, Cell] = {}
        self.rows = 16
        self.cols = 16
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Minesweeper')
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.mainLayout = QVBoxLayout()
        self.centralWidget.setLayout(self.mainLayout)
        
        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(0)
        self.mainLayout.addLayout(self.gridLayout)
        
        numbers = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            ['`', '*', None]
        ]
        
        for row in range(self.rows):
            for col in range(self.cols):
                button = Cell(row, col, logical_dpi)
                button.updateState(CellState.UNREVEALED_EMPTY)
                self.gridLayout.addWidget(button, row, col)
                self.buttons[(row, col)] = button
        
        for row in range(len(numbers)):
            for col in range(len(numbers[row])):
                button = self.buttons[(row, col)]
                if numbers[row][col] is None:
                    button.updateState(CellState.UNREVEALED_EMPTY)
                elif numbers[row][col] == '`': 
                    button.updateState(CellState.UNREVEALED_FLAG)
                elif numbers[row][col] == '*':
                    button.updateState(CellState.REVEALED_MINE)
                elif numbers[row][col] == 0:
                    button.updateState(CellState.REVEALED_EMPTY)
                else:
                    button.updateState(CellState.revealed_num(numbers[row][col]), numbers[row][col])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = max(app.screens(), key=lambda screen: screen.size().width() * screen.size().height())
    logical_dpi = screen.logicalDotsPerInch()
    resolution = screen.size()
    print(logical_dpi, resolution)
    window = TestMineButton()
    window.show()
    sys.exit(app.exec_())