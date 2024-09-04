import random
from enum import Enum

import numpy as np


class CellState(Enum):
    REVEALED_EMPTY = 0
    REVEALED_NUM_1 = 1
    REVEALED_NUM_2 = 2
    REVEALED_NUM_3 = 3
    REVEALED_NUM_4 = 4
    REVEALED_NUM_5 = 5
    REVEALED_NUM_6 = 6
    REVEALED_NUM_7 = 7
    REVEALED_NUM_8 = 8
    REVEALED_MINE = 9
    UNREVEALED_EMPTY = 10
    UNREVEALED_FLAG = 11

    def is_revealed_safe(self):
        return self.value in range(0, 9)

    def is_revealed_num(self):
        return self.value in range(1, 9)

    @staticmethod
    def revealed_num(num):
        assert num in range(1, 9), f'Invalid number {num}!'
        return CellState(num)

    def __eq__(self, other):
        return self.value == other.value


class MinesweeperEnv:
    def __init__(self, rows=16, cols=16, mines=40):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.flags = 0
        self.first_click = True
        self.mine_positions = set()
        self.state = [[CellState.UNREVEALED_EMPTY for _ in range(cols)] for _ in range(rows)]
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.last_updated_cells = set()

    @staticmethod
    def _normalize(value):
        # Normalize value to [0, 1], since flagged will not be used in training, largest value is 10
        return float(value / (len(CellState) - 2))

    def board_to_string(self, mine_position: tuple[int, int] = None, flag_mines=False):
        board_str = ""
        mine_position = mine_position or (-1, -1)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.state[r][c] == CellState.UNREVEALED_FLAG or (flag_mines and self.board[r][c] == -1):
                    board_str += "@ "
                elif self.state[r][c] == CellState.UNREVEALED_EMPTY:
                    board_str += "■ "
                elif self.state[r][c] == CellState.REVEALED_EMPTY:
                    board_str += "□ "
                elif self.state[r][c].is_revealed_num():
                    board_str += f"{self.board[r][c]} "
                elif self.state[r][c] == CellState.REVEALED_MINE or (r, c) == mine_position:
                    board_str += "* "
            board_str += "\n"
        return board_str

    def calculate_adjacent_mines(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == -1:
                    continue
                count = 0
                for r in range(max(0, row-1), min(self.rows, row+2)):
                    for c in range(max(0, col-1), min(self.cols, col+2)):
                        if self.board[r][c] == -1:
                            count += 1
                self.board[row][col] = count

    def check_win(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.state[row][col] == CellState.UNREVEALED_EMPTY and self.board[row][col] != -1:
                    return False
        return True

    def count_flags_around(self, row, col):
        count = 0
        for r in range(max(0, row-1), min(self.rows, row+2)):
            for c in range(max(0, col-1), min(self.cols, col+2)):
                if (r, c) != (row, col) and self.state[r][c] == CellState.UNREVEALED_FLAG:
                    count += 1
        return count

    def flag_cell(self, row, col):
        # TODO meger into make move
        if self.state[row][col] == CellState.UNREVEALED_EMPTY:
            self.state[row][col] = CellState.UNREVEALED_FLAG
            self.flags += 1
        elif self.state[row][col] == CellState.UNREVEALED_FLAG:
            self.state[row][col] = CellState.UNREVEALED_EMPTY
            self.flags -= 1
        self.last_updated_cells.add((row, col))

    def generate_mines(self, first_row, first_col):
        all_positions = set(range(self.rows * self.cols))
        first_click_position = first_row * self.cols + first_col
        all_positions.remove(first_click_position)
        around_cells = self.get_around_cells(first_row, first_col)
        around_positions = set([r * self.cols + c for r, c in around_cells])
        all_positions = all_positions.difference(around_positions)
        self.mine_positions = set(random.sample(list(all_positions), self.mines))
        for pos in self.mine_positions:
            row, col = divmod(pos, self.cols)
            self.board[row][col] = -1
        self.calculate_adjacent_mines()

    def get_around_cells(self, row, col):
        for r in range(max(0, row-1), min(self.rows, row+2)):
            for c in range(max(0, col-1), min(self.cols, col+2)):
                if (r, c) != (row, col):
                    yield r, c

    def get_around_flagged_cells(self, row, col):
        for r, c in self.get_around_cells(row, col):
            if self.state[r][c] == CellState.UNREVEALED_FLAG:
                yield r, c

    def get_around_unrevealed_empty_cells(self, row, col):
        for r, c in self.get_around_cells(row, col):
            if self.state[r][c] == CellState.UNREVEALED_EMPTY:
                yield r, c

    def get_vh_neighbours(self, row, col):
        for r, c in self.get_around_cells(row, col):
            if r == row or c == col:
                yield r, c

    def get_game_state(self):
        return self.state, self.board, self.flags, self.mine_positions

    def get_normalized_state(self):
        return np.array(
            [[self._normalize(self.state[row][col].value) for col in range(self.cols)] for row in range(self.rows)],
            dtype=np.float32)

    def get_valid_actions(self):
        valid_actions = [(row, col) for row in range(self.rows) for col in range(self.cols) if self.state[row][col] == CellState.UNREVEALED_EMPTY]
        return valid_actions

    def is_in_board(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def make_move(self, row, col, allow_click_revealed_num=False, allow_recursive=True, allow_retry=False):
        """
        :return: is game over
        """
        if self.first_click:
            self.first_click = False
            self.generate_mines(row, col)
        if (allow_click_revealed_num and
            self.state[row][col].is_revealed_num() and
            self.count_flags_around(row, col) == self.board[row][col]
        ):
            self.reveal_neighbouring_cells(row, col, allow_recursive)
            return False  # Continue game
        if self.reveal_cell(row, col, allow_recursive):
            # Not mine
            return False # Continue game
        # mine
        if allow_retry:
            self.state[row][col] = CellState.UNREVEALED_EMPTY
        return True  # Game lose

    def new_game(self, rows=None, cols=None, mines=None):
        self.rows = rows or self.rows
        self.cols = cols or self.cols
        self.mines = mines or self.mines
        self.reset()
    
    def reset(self):
        self.flags = 0
        self.first_click = True
        self.mine_positions.clear()
        self.state = [[CellState.UNREVEALED_EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def replay(self):
        self.flags = 0
        self.first_click = False
        self.state = [[CellState.UNREVEALED_EMPTY for _ in range(self.cols)] for _ in range(self.rows)]

    def reveal_cell(self, row, col, allow_recursive=True):
        """
        :return: is safe reveal
        """
        self.last_updated_cells.add((row, col))
        if self.state[row][col] != CellState.UNREVEALED_EMPTY:
            return True
        if self.board[row][col] == 0:
            # not mine
            self.state[row][col] = CellState.REVEALED_EMPTY
            if allow_recursive and self.board[row][col] == 0:
                for r in range(max(0, row-1), min(self.rows, row+2)):
                    for c in range(max(0, col-1), min(self.cols, col+2)):
                        if (r, c) != (row, col):
                            self.reveal_cell(r, c, allow_recursive)
            return True
        elif self.board[row][col] > 0:
            self.state[row][col] = CellState.revealed_num(self.board[row][col])
            return True
        # mine
        self.state[row][col] = CellState.REVEALED_MINE
        return False

    def reveal_neighbouring_cells(self, row, col, allow_recursive=True):
        for r in range(max(0, row-1), min(self.rows, row+2)):
            for c in range(max(0, col-1), min(self.cols, col+2)):
                if (r, c) != (row, col) and self.state[r][c] == CellState.UNREVEALED_EMPTY:
                    self.reveal_cell(r, c, allow_recursive)
