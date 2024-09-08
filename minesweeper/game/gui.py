import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QMessageBox, QLabel, QVBoxLayout, QHBoxLayout, QGraphicsDropShadowEffect, QAction
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QBrush, QPen, QPainter, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal

from .minesweeper_env import MinesweeperEnv, CellState

COLORS = {
    '1': QColor(0, 0, 255),      # 蓝色
    '2': QColor(0, 128, 0),      # 绿色
    '3': QColor(255, 0, 0),      # 红色
    '4': QColor(0, 0, 128),      # 深蓝色
    '5': QColor(128, 0, 0),      # 棕色
    '6': QColor(0, 128, 128),    # 青色
    '7': QColor(0, 0, 0),        # 黑色
    '8': QColor(128, 128, 128),   # 灰色
    '*': QColor(0, 0, 0),        # 黑色
}

class MinesweeperGame(QMainWindow):
    def __init__(self, rows=16, cols=16, mines=40, *, logical_dpi):
        super().__init__()
        self.dpi = logical_dpi
        self.last_action_cells = set()

        self.env = MinesweeperEnv(rows, cols, mines)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Minesweeper')
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.mainLayout = QVBoxLayout()
        self.centralWidget.setLayout(self.mainLayout)
        
        self.topLayout = QGridLayout()
        self.mainLayout.addLayout(self.topLayout)

        self.mineLabel = QLabel(f'{self.env.mines - self.env.flags}')
        self.mineLabel.setFont(QFont('Microsoft Yahei', int(0.09 * self.dpi)))
        self.topLayout.addWidget(self.mineLabel, 0, 0)

        self.restartButton = QPushButton('😊')
        self.restartButton.setFont(QFont('Arial', int(0.09 * self.dpi)))
        self.restartButton.setFixedSize(int(0.3 * self.dpi), int(0.3 * self.dpi))
        self.restartButton.clicked.connect(self.resetGame)
        self.topLayout.addWidget(self.restartButton, 0, 1)

        self.empty = QLabel()
        self.topLayout.addWidget(self.empty, 0, 2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(0)
        self.mainLayout.addLayout(self.gridLayout)

        self.cells: dict[tuple, Cell] = {}
        self.initGame()

        self.createMenuBar()
        
    def createMenuBar(self):
        menubar = self.menuBar()
        gameMenu = menubar.addMenu('Game')

        easyAction = QAction('Easy', self)
        easyAction.triggered.connect(lambda: self.newGame(8, 8, 10))
        gameMenu.addAction(easyAction)

        mediumAction = QAction('Medium', self)
        mediumAction.triggered.connect(lambda: self.newGame(16, 16, 40))
        gameMenu.addAction(mediumAction)

        hardAction = QAction('Hard', self)
        hardAction.triggered.connect(lambda: self.newGame(30, 16, 99))
        gameMenu.addAction(hardAction)

    def add_cell(self, row, col):
        button = Cell(row, col, self.dpi)
        button.leftReleased.connect(self.makeMoveHndlr(row, col, flag=False))
        button.rightClicked.connect(self.makeMoveHndlr(row, col, flag=True))
        self.gridLayout.addWidget(button, row, col)
        self.cells[(row, col)] = button

    def initGame(self):
        print("New Game")
        self.cells = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                self.add_cell(row, col)

    def makeMoveHndlr(self, row, col, flag: bool, show_last_action=False, allow_click_revealed_num=True, allow_recursive=True):
        if flag:
            def handler_flag():
                last_action = (row, col) if show_last_action else None
                self.env.make_move(row, col, flag=True)
                self.updateCells(last_action=last_action)
                self.updateMineLabel()
            return handler_flag

        def handler_reveal():
            last_action = (row, col) if show_last_action else None
            if self.env.make_move(row, col, flag=False, allow_click_revealed_num=allow_click_revealed_num, allow_recursive=allow_recursive):
                self.revealAllMines()
                self.updateCells(last_action=last_action)
                self.gameOver(False)
                return
            elif self.env.check_win():
                self.updateCells(last_action=last_action)
                self.gameOver(True)
                return
            self.updateCells(last_action=last_action)
        return handler_reveal

    def updateCell(self, row, col):
        state, board, flags, mine_positions = self.env.get_game_state()
        cell = self.cells[(row, col)]
        cell.updateState(state[row][col], mines_around=board[row][col])

    def updateCells(self, last_action=None):
        state, board, flags, mine_positions = self.env.get_game_state()
        to_be_updated = self.env.last_updated_cells.union(self.last_action_cells)
        self.env.last_updated_cells.clear()
        self.last_action_cells.clear()
        for row, col in to_be_updated:
            cell = self.cells[(row, col)]
            is_last_action = (row, col) == last_action
            if is_last_action:
                self.last_action_cells.add((row, col))
            cell.updateState(state[row][col], mines_around=board[row][col], is_last_action=is_last_action)

    def revealAllMines(self):
        _, _, _, mine_positions = self.env.get_game_state()
        for pos in mine_positions:
            row, col = divmod(pos, self.env.cols)
            if self.env.state[row][col] != CellState.UNREVEALED_FLAG:
                self.env.reveal_cell(row, col)

    def gameOver(self, won):
        if won:
            if QMessageBox.information(self, "Game over", "You won!\nOK: New game, Cancel: Replay", QMessageBox.Ok|QMessageBox.Cancel, QMessageBox.Ok) == QMessageBox.Ok:
                self.resetGame()
            else:
                self.replayGame()
        else:
            if QMessageBox.information(self, "Game over", "Game Over!\nOK: New game, Cancel: Replay", QMessageBox.Ok|QMessageBox.Cancel, QMessageBox.Ok) == QMessageBox.Ok:
                self.resetGame()
            else:
                self.replayGame()

    def newGame(self, rows, cols, mines):
        old_rows, old_cols = self.env.rows, self.env.cols
        new_rows, new_cols = rows, cols

        cells_add = []
        cells_delete = []

        if new_rows > old_rows:
            for row in range(old_rows, new_rows):
                for col in range(old_cols):
                    cells_add.append((row, col))
        if new_cols > old_cols:
            for row in range(new_rows):
                for col in range(old_cols, new_cols):
                    cells_add.append((row, col))

        if new_rows < old_rows:
            for row in range(new_rows, old_rows):
                for col in range(old_cols):
                    cells_delete.append((row, col))
        if new_cols < old_cols:
            for row in range(new_rows):
                for col in range(new_cols, old_cols):
                    cells_delete.append((row, col))

        for row in range(min(new_rows, old_rows)):
            for col in range(min(new_cols, old_cols)):
                self.cells.get((row, col)).updateState(CellState.UNREVEALED_EMPTY)

        for row, col in cells_add:
            self.add_cell(row, col)

        for row, col in cells_delete:
            button = self.cells.pop((row, col))
            self.gridLayout.removeWidget(button)
            button.deleteLater()

        self.env.new_game(rows, cols, mines)
        self.updateMineLabel()

    def replayGame(self):
        self.env.replay()
        self.updateMineLabel()
        for cell in self.cells.values():
            cell.updateState(CellState.UNREVEALED_EMPTY)

    def resetGame(self):
        self.env.reset()
        self.updateMineLabel()
        for cell in self.cells.values():
            cell.updateState(CellState.UNREVEALED_EMPTY)

    def updateMineLabel(self):
        self.mineLabel.setText(f'{self.env.mines - self.env.flags}')


class Cell(QPushButton):
    leftReleased = pyqtSignal()
    rightClicked = pyqtSignal()

    def __init__(self, row, col, dpi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row = row
        self.col = col
        self.state = CellState.UNREVEALED_EMPTY
        self.text = ''
        self.dpi = dpi
        # 3840 x 2160: 32 x 32
        # 1920 x 1080: 16 x 16
        self.setFixedSize(int(0.22 * self.dpi), int(0.22 * self.dpi))
        self.setFont(QFont('MINE-SWEEPER'))
        self._updateStyle()
        self.is_pressed = False
        # 创建阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(int(0.07 * self.dpi))
        self.shadow.setColor(QColor(0, 0, 0))
        self.shadow.setOffset(0, 0)
        self.setGraphicsEffect(self.shadow)

    def mousePressEvent(self, event):
        if self.rect().contains(event.pos()):
            if event.button() == Qt.LeftButton and self.state != CellState.UNREVEALED_FLAG:
                self.is_pressed = True
                style = self._getStyle(holding=True)
                self.setStyleSheet(style)
            elif event.button() == Qt.RightButton:
                self.rightClicked.emit()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect().contains(event.pos()):
            if event.button() == Qt.LeftButton and self.state != CellState.UNREVEALED_FLAG:
                if self.is_pressed:
                    self.leftReleased.emit()
                self.is_pressed = False
        else:
            super().mouseReleaseEvent(event)
        self._updateStyle()
        
    def updateState(self, new_state, mines_around=None, is_last_action=None):
        self.state = new_state
        if self.state == CellState.UNREVEALED_FLAG:
            self.text = '`'
        elif self.state.is_revealed_num():
            self.text = str(mines_around)
        elif self.state == CellState.REVEALED_MINE:
            self.text = '*'
        else:
            self.text = ''
        self._updateStyle(is_last_action=is_last_action)

    def _getStyle(self, holding=False, is_last_action=None):
        nudge = {
            '1': 0.0035 * self.dpi,
            '*': 0.0035 * self.dpi,
            '`': 0.007 * self.dpi,
        }
        border = 0 * self.dpi
        # 3840 x 2160: 40 x 40
        # 1920 x 1080:  x
        font_normal = f' font-size: {int(0.14 * self.dpi)}px;'
        font_flag = f' font-size: {int(0.16 * self.dpi)}px;'
        font_mine = f' font-size: {int(0.14 * self.dpi)}px;'
        csstext = f'border: {border}px solid gray;'
        color = COLORS.get(self.text, QColor(0, 0, 0))
        if self.state == CellState.REVEALED_EMPTY or self.state.is_revealed_num():
            background_color = 'darkgray' if holding else 'lightgoldenrodyellow' if not is_last_action else 'yellow'
            csstext += font_normal
            csstext += f' color: {color.name()}; background-color: {background_color};'
            csstext += f' padding: 0px -{nudge.get(self.text, 0)}px 0px {nudge.get(self.text, 0)}px;'
        elif self.state == CellState.REVEALED_MINE:
            background_color = 'red' if not is_last_action else 'yellow'
            csstext += font_mine
            csstext += f' color: {color.name()}; background-color: {background_color}; padding: 0px -{nudge[self.text]}px 0px {nudge[self.text]}px;'
        elif self.state == CellState.UNREVEALED_FLAG:
            csstext += font_flag
            background_color = 'darkgray' if holding else 'lightgray' if not is_last_action else 'yellow'
            csstext += f' background-color: {background_color}; padding: 0px -{nudge[self.text]}px 0px {nudge[self.text]}px;'
        else:
            background_color = 'darkgray' if holding else 'lightgray' if not is_last_action else 'yellow'
            csstext += f' background-color: {background_color}; padding: 0px;'
        return csstext

    def _updateStyle(self, is_last_action=None):
        self.setStyleSheet(self._getStyle(is_last_action=is_last_action))
        self.setText(self.text)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.state == CellState.UNREVEALED_FLAG:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # 创建渐变效果
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0.0, QColor(255, 0, 0))  # 红色
            gradient.setColorAt(0.50, QColor(255, 0, 0))  # 红色
            gradient.setColorAt(0.61, QColor(0, 0, 0))  # 黑色
            gradient.setColorAt(1.0, QColor(0, 0, 0))  # 黑色

            # 设置渐变画刷
            brush = QBrush(gradient)
            painter.setPen(QPen(brush, 1))

            rect = self.rect()
            font_metrics = QFontMetrics(self.font())
            text_rect = font_metrics.boundingRect(rect, Qt.AlignCenter, self.text)

            # 确保文本居中对齐
            painter.drawText(rect, Qt.AlignCenter, self.text)
