from game.minesweeper_env import MinesweeperEnv, CellState
from solver.classic.coordtransformer import CoordTransformer


class NumCouple:
    def __init__(self, r1, c1, r2, c2, env: MinesweeperEnv):
        self.r1 = min(r1, r2)
        self.c1 = min(c1, c2)
        self.r2 = max(r1, r2)
        self.c2 = max(c1, c2)
        self.env = env
        self.n1 = int(env.state[r1][c1].value)
        self.n2 = int(env.state[r2][c2].value)
        self.t = CoordTransformer(self.r1, self.c1, self.r2, self.c2)

        self.safe_reveals = None
        self.mine_flags = None

    def __eq__(self, other):
        return (self.r1, self.c1) == (other.r1, other.c1) and (self.r2, self.c2) == (other.r2, other.c2)

    def __hash__(self):
        return hash((self.r1, self.c1, self.r2, self.c2))

    def __str__(self):
        return f"({self.r1},{self.c1}):{self.n1}, ({self.r2},{self.c2}):{self.n2}"

    def determine(self):
        # b那一侧的私有unreveal格 == 两个数字之差delta，则b侧有delta个雷，a侧全是safe，公共区域有a个雷
        # TODO 如果大数那一侧的私有unreveal格 > 两个数字之差delta，则b侧有delta~b个雷，a侧和公共区域（即a周围）有a个雷


        coord_a = (1, 1)
        coord_b = (2, 1)
        coords_a_side = [(0,0), (0,1), (0,2)]
        coords_b_side = [(3,0), (3,1), (3,2)]
        coords_common = [(1,0), (1,1), (1,2), (2,2)]

        coords_unrevealed_a_side = []
        coords_unrevealed_b_side = []
        coords_unrevealed_common = []

        # 周围每有一个旗，数字减1
        flag_around_a = self.env.count_flags_around(self.r1, self.c1)
        flag_around_b = self.env.count_flags_around(self.r2, self.c2)

        a = self.n1 - flag_around_a
        b = self.n2 - flag_around_b
        # 默认a小b大，如果不是，交换
        if a > b:
            a, b = b, a
            coord_a, coord_b = coord_b, coord_a
            coords_a_side, coords_b_side = coords_b_side, coords_a_side

        delta = b - a

        # 棋盘外当作已翻开处理
        for coord_a_side in coords_a_side:
            r, c = self.t.v2p(*coord_a_side)
            if not self.env.is_in_board(r, c) or self.env.state[r][c] == CellState.UNREVEALED_EMPTY:
                coords_unrevealed_a_side.append(coord_a_side)

        for coord_b_side in coords_b_side:
            r, c = self.t.v2p(*coord_b_side)
            if not self.env.is_in_board(r, c) or self.env.state[r][c] == CellState.UNREVEALED_EMPTY:
                coords_unrevealed_b_side.append(coord_b_side)

        for coord_common in coords_common:
            r, c = self.t.v2p(*coord_common)
            if not self.env.is_in_board(r, c) or self.env.state[r][c] == CellState.UNREVEALED_EMPTY:
                coords_unrevealed_common.append(coord_common)

        self.safe_reveals = set()
        self.mine_flags = set()
        if len(coords_unrevealed_b_side) == delta:
            for coord in coords_unrevealed_b_side:
                r, c = self.t.v2p(*coord)
                if self.env.is_in_board(r, c):
                    self.mine_flags.add((r, c))
            for coord in coords_unrevealed_a_side:
                r, c = self.t.v2p(*coord)
                if self.env.is_in_board(r, c):
                    self.safe_reveals.add((r, c))
            return True
        return False

    def make_moves(self):
        for r, c in self.safe_reveals:
            yield r, c, False
        for r, c in self.mine_flags:
            yield r, c, True
