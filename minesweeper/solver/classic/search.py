import random

from minesweeper.game.minesweeper_env import MinesweeperEnv, CellState
from solver.classic.numcouple import NumCouple
from solver.classic.unrevealedarea import UnrevealedArea


class ClassicMinesweeperSolver:
    def __init__(self, env: MinesweeperEnv):
        self.env = env
        self.first_click = True
        self.last_seed = None
        self.reset()

    def _has_revealed_num_neighbor(self, row, col):
        for r, c in self.env.get_around_cells(row, col):
            if self.env.state[r][c].is_revealed_num():
                return True
        return False

    def _has_unrevealed_empty_neighbor(self, row, col):
        for r, c in self.env.get_around_cells(row, col):
            if self.env.state[r][c] == CellState.UNREVEALED_EMPTY:
                return True
        return False

    def _get_unrevealed_empty_neighbors(self, row, col):
        for r, c in self.env.get_around_cells(row, col):
            if self.env.state[r][c] == CellState.UNREVEALED_EMPTY:
                yield r, c

    def _get_neighbors_stats(self, row, col):
        for r, c in self.env.get_around_cells(row, col):
            yield r, c, self.env.state[r][c], self.env.board[r][c]

    def reset(self):
        self.first_click = True
        self.last_seed = random.randint(0, 100000)
        # self.last_seed = 56266
        random.seed(self.last_seed)
        print(f"Seed: {self.last_seed}")
        self.safe_reveals = set()
        self.mine_flags = set()

    def replay(self):
        self.first_click = True
        random.seed(self.last_seed)
        print(f"Seed: {self.last_seed}")
        self.safe_reveals = set()
        self.mine_flags = set()

    def update_knowledge_base(self):
        # should perform for all cells again, not just the last one
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.state[row][col] == CellState.UNREVEALED_FLAG:
                    self.mine_flags.discard((row, col))
                if self.env.state[row][col].is_revealed_safe():
                    self.safe_reveals.discard((row, col))
                    self.update_safe_reveals(row, col)
                    self.update_mine_flags(row, col)

    def update_safe_reveals(self, row, col):
        num = self.env.board[row][col]
        unrevealed_neighbors = self.env.get_around_unrevealed_empty_cells(row, col)
        flagged_neighbors = list(self.env.get_around_flagged_cells(row, col))

        if len(flagged_neighbors) == num:
            for r, c in unrevealed_neighbors:
                self.safe_reveals.add((r, c))

    def update_mine_flags(self, row, col):
        num = self.env.board[row][col]
        unrevealed_neighbors = list(self.env.get_around_unrevealed_empty_cells(row, col))
        flagged_neighbors = list(self.env.get_around_flagged_cells(row, col))

        if len(unrevealed_neighbors) + len(flagged_neighbors) == num:
            for r, c in unrevealed_neighbors:
                self.mine_flags.add((r, c))

    def make_random_move(self):
        # TODO 单个格子算概率，选最小的。e.g. Seed: 55865 after Solver advanced 1 move: 36, 7
        row, col = random.choice(self.env.get_valid_actions())
        return row, col, False

    def make_safe_moves(self):
        flag = False
        if self.first_click:
            self.first_click = False
            row, col = random.choice(self.env.get_valid_actions())
            yield row, col, flag
        while self.mine_flags:
            row, col = self.mine_flags.pop()
            yield row, col, True
        while self.safe_reveals:
            row, col = self.safe_reveals.pop()
            yield row, col, False

    # TODO 利用剩余雷数信息
    def make_advanced1_moves(self):
        # 对于一个数num，限定num个雷的区域，然后判断对周围其他数字的影响。
        # TODO 当然也可以有连锁反应，多个不重合的雷限定区域一起组合，再确定能影响的数字，不断迭代。
        # TODO 而且同时考虑多个数字可以扩大雷的限定区域，不断迭代。如两个格时的1、2定律（暂实现为advanced2）
        # TODO 考虑三个格的情况 See TODO.png, https://minesweeper.cn/doc/jichudingshi.htm 最后一条
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.state[row][col].is_revealed_num():
                    unrevealed_points = []
                    flags_around = 0
                    for r, c, state, num in self._get_neighbors_stats(row, col):
                        if state == CellState.UNREVEALED_EMPTY:
                            unrevealed_points.append((r, c))
                        elif state == CellState.UNREVEALED_FLAG:
                            flags_around += 1

                    if 4 >= len(unrevealed_points) >= 2:
                        ua = UnrevealedArea(self.env, unrevealed_points, self.env.board[row][col] - flags_around)
                        if ua.check_affected_revealed_num_neighbors():
                            yield from ua.make_moves()

    def make_advanced2_moves(self):
        """两个相邻数字时的定律"""
        couples: set[NumCouple] = set()
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.state[row][col].is_revealed_num() and self._has_unrevealed_empty_neighbor(row, col):
                    for row1, col1 in self.env.get_vh_neighbours(row, col):
                        if self.env.state[row1][col1].is_revealed_num() and self._has_unrevealed_empty_neighbor(row1, col1):
                            couples.add(NumCouple(row, col, row1, col1, self.env))
        # 每个couple都是两个数周围都有未翻开的空格的数字
        for couple in couples:
            if couple.determine():
                yield from couple.make_moves()
