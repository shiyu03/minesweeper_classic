from collections.abc import Generator

from game.minesweeper_env import MinesweeperEnv


class UnrevealedArea:
    def __init__(self, env: MinesweeperEnv, unrevealed_points: list, mines: int):
        """给定一系列未翻开格子的位置，寻找能被这些格子同时影响到的已翻开的格子
        """
        self.env = env
        self.unrevealed_points = unrevealed_points
        self.mines = mines
        self.safe_reveals = None
        self.mine_flags = None

    @staticmethod
    def _is_around(p1, p2):
        return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1

    def _possible_grid_centers(self, points) -> Generator[tuple[int, int]]:
        """同时受所有未翻开格子影响的3x3区域的中心点（所有）。方法：递归+排除法。"""
        l = len(points)
        if l < 1:
            return []
        if l == 1:
            r, c = points[0]
            yield from self.env.get_around_cells(r, c)
        else:
        # if len(unrevealed_points) == 2:
        #     for center in self._possible_grid_centers([unrevealed_points[0]]):
        #         if self._is_around(center, unrevealed_points[1]) and center != unrevealed_points[1]:
        #             yield center
            for center in self._possible_grid_centers(points[:l - 1]):
                if self._is_around(center, points[l - 1]) and center != points[l - 1]:
                    yield center

    def _affected_revealed_num_neighbors(self):
        """除去excepted_centers的其他受影响的数字格子"""
        affected_revealed_num_cells = set()
        for r, c in self._possible_grid_centers(self.unrevealed_points):
            if self.env.state[r][c].is_revealed_num():
                affected_revealed_num_cells.add((r, c))
        return affected_revealed_num_cells

    def check_affected_revealed_num_neighbors(self):
        """确定2<=n<=3个雷所在的限定区域后，对于受该区域影响的所有已翻开数字num，如果num-n==0，它的四周除去限定区域全safe; 如果num-n==除去限定区域后的格数，则全是雷"""
        # 可以把所有的ua记录下来，动态涨消，每次遍历一遍检查有没有符合条件的。
        self.safe_reveals = set()
        self.mine_flags = set()
        bingo = False
        for r, c in self._affected_revealed_num_neighbors():
            other_unrevealed_neighbors = set(self.env.get_around_unrevealed_empty_cells(r, c)) - set(self.unrevealed_points)
            if self.env.board[r][c] - self.env.count_flags_around(r, c)  - self.mines == 0:
                self.safe_reveals.update(other_unrevealed_neighbors)
                bingo = True
            elif self.env.board[r][c] - self.env.count_flags_around(r, c) - self.mines == len(other_unrevealed_neighbors):
                self.mine_flags.update(other_unrevealed_neighbors)
                bingo = True
        return bingo

    def make_moves(self):
        for r, c in self.safe_reveals:
            yield r, c, False
        for r, c in self.mine_flags:
            yield r, c, True
