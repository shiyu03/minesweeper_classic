from PyQt5.QtWidgets import QMainWindow, QPushButton, QGridLayout, QWidget, QMessageBox, QLabel, QVBoxLayout, QGraphicsDropShadowEffect, QAction
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QBrush, QPen, QPainter, QFontMetrics, QIntValidator
from PyQt5.QtCore import Qt, pyqtSignal

from .minesweeper_env import MinesweeperEnv, CellState
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QVBoxLayout

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
        self.is_game_lose = False
        self.is_game_win = False
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
        self.mineLabel.setFont(QFont('Microsoft Yahei', int(0.12 * self.dpi)))
        self.topLayout.addWidget(self.mineLabel, 0, 0)

        self.restartButton = QPushButton('😊')
        self.restartButton.setFont(QFont('Arial', int(0.15 * self.dpi)))
        self.restartButton.setFixedSize(int(0.33 * self.dpi), int(0.33 * self.dpi))
        self.restartButton.clicked.connect(lambda: [self.resetGame(), self.updateRestartButtonText()])
        self.topLayout.addWidget(self.restartButton, 0, 1)

        self.empty = QLabel()
        self.topLayout.addWidget(self.empty, 0, 2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(0)
        self.mainLayout.addLayout(self.gridLayout)

        self.cells: dict[tuple, Cell] = {}
        self.firstGame()

        self.createMenuBar()
        self.updateWindowSize()

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

        customAction = QAction('Custom', self)
        customAction.triggered.connect(self.customGame)
        gameMenu.addAction(customAction)

    def customGame(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Custom Game')

        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        rows_input = QLineEdit(dialog)
        rows_input.setValidator(QIntValidator(1, 50, dialog))
        form_layout.addRow('rows(1,50):', rows_input)

        cols_input = QLineEdit(dialog)
        cols_input.setValidator(QIntValidator(1, 50, dialog))
        form_layout.addRow('columns(1,50):', cols_input)

        mines_input = QLineEdit(dialog)
        mines_input.setValidator(QIntValidator(dialog))
        form_layout.addRow('mines:', mines_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(lambda: self.validateInput(rows_input, cols_input, mines_input, dialog))
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec_()

    def validateInput(self, rows_input, cols_input, mines_input, dialog):
        try:
            rows = int(rows_input.text())
            cols = int(cols_input.text())
            mines = int(mines_input.text())

            if not (1 <= rows <= 50) or not (1 <= cols <= 50) or not (1 <= mines < rows*cols):
                raise ValueError("Input out of range")

            # 如果所有输入都有效，调用 newGame 并关闭对话框
            self.newGame(rows, cols, mines)
            dialog.accept()

        except ValueError as e:
            # 显示错误消息框
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText(str(e) + "\nPlease enter values again.")
            msg_box.exec_()

    def add_cell(self, row, col):
        button = Cell(row, col, self.dpi)
        button.leftDown.connect(lambda: self.updateRestartButtonText(is_keydown=True))
        button.leftAvailReleased.connect(self.makeMoveHndlr(row, col, flag=False))
        button.rightDown.connect(lambda: [self.makeMoveHndlr(row, col, flag=True)(), self.updateRestartButtonText(is_keydown=True)])
        button.mouseReleased.connect(lambda: self.updateRestartButtonText())
        self.gridLayout.addWidget(button, row, col)
        self.cells[(row, col)] = button

    @property
    def is_game_over(self):
        return self.is_game_win or self.is_game_lose

    def makeMoveHndlr(self, row, col, flag: bool, show_last_action=False, allow_click_revealed_num=True, allow_recursive=True):
        if flag:
            def handler_flag():
                if self.is_game_over:
                    return
                last_action = (row, col) if show_last_action else None
                self.env.make_move(row, col, flag=True)
                self.updateCells(last_action=last_action)
                self.updateMineLabel()
            return handler_flag

        def handler_reveal():
            if self.is_game_over:
                return
            last_action = (row, col) if show_last_action else None
            lose, _ = self.env.make_move(row, col, flag=False, allow_click_revealed_num=allow_click_revealed_num, allow_recursive=allow_recursive)
            if lose:
                self.is_game_lose = True
                self.revealAllMines()
                self.updateCells(last_action=last_action)
                self.gameOver(False)
                return
            elif self.env.check_win():
                self.is_game_win = True
                self.updateCells(last_action=last_action)
                self.gameOver(True)
                return
            self.updateCells(last_action=last_action)
        return handler_reveal

    # def updateCell(self, row, col):
    #     state, board, flags, mine_positions = self.env.get_game_state()
    #     cell = self.cells[(row, col)]
    #     cell.updateState(state[row][col], mines_around=board[row][col])

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
                self.env.reveal_cell(row, col, set())

    def gameOver(self, won):
        self.updateRestartButtonText()
        # if won:
        #     rst = QMessageBox.information(self, "Game over", "You won!\nOK: New game, Cancel: Replay",
        #                             QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        #     if rst == QMessageBox.Ok:
        #         self.resetGame()
        #     elif rst == QMessageBox.Cancel:
        #         self.replayGame()

    def firstGame(self):
        print("Start Game")
        self.is_game_lose = False
        self.is_game_win = False
        self.cells = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                self.add_cell(row, col)

    def newGame(self, rows, cols, mines):
        print("New Game")
        self.is_game_lose = False
        self.is_game_win = False
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
                self.cells[(row, col)].updateState(CellState.UNREVEALED_EMPTY)

        for row, col in cells_add:
            self.add_cell(row, col)

        for row, col in cells_delete:
            button = self.cells.pop((row, col))
            self.gridLayout.removeWidget(button)
            button.cleanup()
            button.deleteLater()

        self.env.new_game(rows, cols, mines)
        self.updateMineLabel()
        self.updateWindowSize()

    def replayGame(self):
        self.is_game_lose = False
        self.is_game_win = False
        self.env.replay()
        self.updateMineLabel()
        for cell in self.cells.values():
            cell.updateState(CellState.UNREVEALED_EMPTY)

    def resetGame(self):
        self.is_game_lose = False
        self.is_game_win = False
        self.env.reset()
        self.updateMineLabel()
        for cell in self.cells.values():
            cell.updateState(CellState.UNREVEALED_EMPTY)

    def updateMineLabel(self):
        self.mineLabel.setText(f'{self.env.mines - self.env.flags}')

    def updateWindowSize(self):
        self.gridLayout.invalidate()
        self.gridLayout.activate()
        self.updateGeometry()
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

    def initKeyPressListener(self):
        self.centralWidget.setFocusPolicy(Qt.StrongFocus)
        self.centralWidget.keyPressEvent = self.keyPressEvent

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.restartButton.click()

    def updateRestartButtonText(self, *, is_keydown=False):
        if self.is_game_win:
            self.restartButton.setText('😎')
        elif self.is_game_lose:
            self.restartButton.setText('😵')
        elif is_keydown:
            self.restartButton.setText('😲')
        else:
            self.restartButton.setText('😊')

class Cell(QPushButton):
    leftDown = pyqtSignal()
    leftAvailReleased = pyqtSignal()
    rightDown = pyqtSignal()
    mouseReleased = pyqtSignal()

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
        # 创建阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(int(0.07 * self.dpi))
        self.shadow.setColor(QColor(0, 0, 0))
        self.shadow.setOffset(0, 0)
        self.setGraphicsEffect(self.shadow)
        self._plot_qvalue = False
        self._radius = None
        self._rgb_tuple = None

    def cleanup(self):
        self.setGraphicsEffect(None)

    def mousePressEvent(self, event):
        if self.rect().contains(event.pos()):
            if event.button() == Qt.LeftButton and self.state != CellState.UNREVEALED_FLAG:
                self.leftDown.emit()
                self._updateStyle(holding=True)
            elif event.button() == Qt.RightButton:
                self.rightDown.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect().contains(event.pos()):
            if event.button() == Qt.LeftButton and self.state != CellState.UNREVEALED_FLAG:
                self.leftAvailReleased.emit()
        self.mouseReleased.emit()
        self._updateStyle()
        super().mouseReleaseEvent(event)

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
            self._plot_qvalue = False
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

    def _updateStyle(self, holding=False, is_last_action=None):
        if holding:
            self.setStyleSheet(self._getStyle(holding=True))
        else:
            self.setStyleSheet(self._getStyle(is_last_action=is_last_action))
            self.setText(self.text)

    def plotQValue(self, rgb_tuple, radius):
        self._plot_qvalue = True
        self._rgb_tuple = rgb_tuple
        self._radius = radius
        self.update()

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
        elif self.state == CellState.UNREVEALED_EMPTY and self._plot_qvalue:
            # Draw the circle on top of the button
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor(*self._rgb_tuple)))
            painter.setPen(Qt.NoPen)
            center = self.rect().center()
            painter.drawEllipse(center, self._radius, self._radius)