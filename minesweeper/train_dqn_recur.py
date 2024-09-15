import os
import sys

from train_dqn import ExperimentNormal

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from minesweeper.game.minesweeper_env import CellState


class ExperimentNormalRecur(ExperimentNormal):
    def _calc_reward(self, action, failed_cnt, reveald_cnt, skipped_cells: set):
        row, col = action
        lose, last_revealed_cells = self.env.make_move(row, col, flag=False, allow_click_revealed_num=False,
                                                       allow_recursive=True, allow_retry=self.allow_retry)
        if lose:
            failed_cnt += 1
            done = True
            reward = -self.agent.MAX_REWARD
            # 周围没有翻开的格子时，惩罚更大，避免乱猜雷
            if all(self.env.state[r][c] == CellState.UNREVEALED_EMPTY for r, c in self.env.get_around_cells(row, col)):
                penalty = self.agent.MAX_REWARD * 0.5
                reward -= penalty
        else:
            reveald_cnt += len(last_revealed_cells)
            done = self.env.check_win()

            # 开的越多，奖励越多
            explore_prcnt = reveald_cnt / (self.env.rows * self.env.cols - self.env.mines)
            reward = self.agent.MAX_REWARD * explore_prcnt

        # 错的次数越多，当前盘面越难，尽量避免
        # TODO 这个规则可以后期再微调，前期可以关掉
        # difficulty_penalty_prcnt = map_range(failed_cnt, self.MAX_RETRY, 0.2, in_min=0, out_min=0)
        # reward -= self.agent.MAX_REWARD * difficulty_penalty_prcnt
        return reward, done, lose, reveald_cnt, failed_cnt

    def _test_play(self):
        win_rate, avg_revealed_prcnt = self.agent.test_play(self.env, test_episodes=100, allow_recursive=True)
        print(f"Test: Win rate: {win_rate:.2f}, Avg revealed%: {avg_revealed_prcnt:.2f}")


if __name__ == '__main__':
    exp = ExperimentNormalRecur()
    exp.run()

