import os
import sys

from train_dqn import ExperimentNormal
from train_dqn_wflag import ExperimentWFlag

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from minesweeper.game.minesweeper_env import CellState


class ExperimentWFlagRecur(ExperimentWFlag):
    def _calc_reward(self, action, failed_cnt, reveald_cnt, flagged_cnt, valid_actions_flag, valid_actions_reveal, skipped_cells: set):
        row, col, flag = action
        # avoid wrong action made again.
        if flag == 1:
            valid_actions_flag.remove((row, col, 1))
            lose, _ = self.env.make_move(row, col, flag=True, allow_click_revealed_num=False,
                                         allow_recursive=True,
                                         allow_retry=self.allow_retry)
            has_num_around = any(self.env.is_neighbor(row, col, r, c) for r, c in self.env.get_around_cells(row, col))
            if lose or has_num_around:
                # 如果flag错了直接结束，防止影响后续策略
                # flag对了也不能乱打，否则也算输
                done = True
                reward = -self.agent.MAX_REWARD
            else:
                # 但是flag对了比开雷的奖励还要大，防止agent不flag
                flagged_cnt += 1
                done = False
                reward = self.agent.MAX_REWARD
                try:
                    valid_actions_reveal.remove((row, col, 0))
                except ValueError:
                    # 上一把试错踩雷已经去掉了这个格子
                    pass

        else:
            lose, last_revealed_cells = self.env.make_move(row, col, flag=False, allow_click_revealed_num=False,
                                         allow_recursive=True,
                                         allow_retry=self.allow_retry)
            valid_actions_reveal.remove((row, col, 0))
            if lose:
                failed_cnt += 1
                done = True
                reward = -self.agent.MAX_REWARD
                # 周围没有翻开的格子时，惩罚更大，避免乱猜雷
                if all(self.env.state[r][c] == CellState.UNREVEALED_EMPTY for r, c in
                       self.env.get_around_cells(row, col)):
                    penalty = self.agent.MAX_REWARD * 0.5
                    reward -= penalty
            else:
                try:
                    valid_actions_flag.remove((row, col, 1))
                except ValueError:
                    # 上一把flag错了已经去掉了这个格子
                    pass
                reveald_cnt += len(last_revealed_cells)
                done = self.env.check_win()

                # 开的越多，奖励越多
                explore_prcnt = (reveald_cnt + flagged_cnt) / (self.env.rows * self.env.cols)
                reward = self.agent.MAX_REWARD * explore_prcnt

            # 错的次数越多，当前盘面越难，尽量避免
            # TODO 这个规则可以后期再微调，前期可以关掉
            # difficulty_penalty_prcnt = map_range(failed_cnt, self.MAX_RETRY, 0.2, in_min=0, out_min=0)
            # reward -= self.agent.MAX_REWARD * difficulty_penalty_prcnt
        return reward, done, lose, reveald_cnt, failed_cnt, flagged_cnt

    def _test_play(self):
        win_rate, avg_revealed_prcnt, avg_flag_prcnt = self.agent.test_play(self.env, test_episodes=100, allow_recursive=True)
        print(
            f"Test: Win rate: {win_rate:.2f}, Avg revealed%: {avg_revealed_prcnt:.2f}, Avg correct flag%: {avg_flag_prcnt:.2f}")

if __name__ == '__main__':
    exp = ExperimentWFlagRecur()
    exp.run()

