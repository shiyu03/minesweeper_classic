import argparse
import os
import sys
import time
from abc import abstractmethod

import numpy as np
import torch

from solver.dqn_agent import DQNAgent

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from minesweeper.game.minesweeper_env import MinesweeperEnv, CellState
from minesweeper.const import ROOT

def map_range(value, in_max, out_max, in_min=0.0, out_min=0.0):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class Experiment:
    def __init__(self):
        self.args = self._parse_args()
        self.env = None
        self.agent = None
        self.save_every = None
        self.checkpoint_dir = None
        self.batch_size = None
        self.allow_retry = None
        self.MAX_RETRY = None
        self.train_every = None
        self.report_every = None
        self.test_every = None
        self.best_win_rate = 0.0
        self.setup()

    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--rows', default=6, type=int, help='Number of rows in the Minesweeper grid')
        parser.add_argument('-c', '--cols', default=6, type=int, help='Number of columns in the Minesweeper grid')
        parser.add_argument('-m', '--mines', default=5, type=int, help='Number of mines in the Minesweeper grid')
        parser.add_argument('-e', '--eval', action='store_true', help='Evaluate the agent')
        parser.add_argument('--ckpt', default=os.path.join(ROOT, "checkpoint"), help='Checkpoint path')
        parser.add_argument('--log_suffix', default="_normal", help='suffix for log dirname')
        parser.add_argument('--reseteps', action='store_true', help='reset epsilon')
        parser.add_argument('--ba', default=64, type=int, help='Batch size')
        parser.add_argument('--save_every', default=5000, type=int, help='Save checkpoint every n episodes')
        parser.add_argument('--test_every', default=1000, type=int, help='Test play every n episodes')
        parser.add_argument('--update_model_structure', action='store_true', help='enable after first change model\'s structure. Don\'t forget to change the mapping in update_checkpoint()')
        parser.add_argument('--freeze_old_layers', action='store_true', help='Whether or not train new layers only if model\'s structure is updated.')
        args = parser.parse_args()
        return args

    @abstractmethod
    def setup(self): ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def _calc_reward(self, *args): ...

    @abstractmethod
    def _test_play(self, test_episodes): ...

    def print_board(self, mine_position):
        # 打印棋盘状态
        row, col = mine_position
        sys.stdout.write("\033c")  # 清除终端
        sys.stdout.write('\n')
        sys.stdout.write(f"Reveal: ({row}, {col}) - {self.env.state[row][col].name}\n")
        board_str = self.env.board_to_string(mine_position, flag_mines=True)
        sys.stdout.write(board_str)


class ExperimentNormal(Experiment):
    def __init__(self):
        super().__init__()

    def setup(self):
        self.env = MinesweeperEnv(rows=self.args.rows, cols=self.args.cols, mines=self.args.mines)
        self.agent = DQNAgent(self.args.log_suffix, self.args.eval)

        self.save_every = self.args.save_every
        self.checkpoint_dir = self.args.ckpt
        os.makedirs(self.args.ckpt, exist_ok=True)
        self.batch_size = self.args.ba
        self.allow_retry = True if not self.args.eval else False
        self.MAX_RETRY = self.args.mines if self.allow_retry else 1
        self.train_every = 10
        self.report_every = 1000
        self.test_every = self.args.test_every
        self.best_win_rate = 0.0

        # 尝试加载检查点
        try:
            self.agent.load(self.checkpoint_dir, self.args.update_model_structure, self.args.freeze_old_layers)
            self.best_win_rate = self.agent.best_winrate
            if self.args.reseteps:
                self.agent.epsilon = 1.0
                print("WARNING: Reset epsilon to 1.0!")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        if self.args.eval:
            self.agent.model.eval()

    def run(self):
        while True:
            self.env.reset()
            self.agent.episodes += 1
            self._train_episode()

            if self.agent.episodes % self.save_every == 0:
                checkpoint_filename = os.path.join(
                    self.checkpoint_dir,
                    f"dqn_{self.env.rows}x{self.env.cols}x{self.env.mines}_ep{self.agent.episodes}_ba{self.batch_size}.pth")
                self.agent.save(checkpoint_filename)
                print(f"Checkpoint saved at episode {self.agent.episodes}.")

            if self.test_every > 0 and self.agent.episodes % self.test_every == 0:
                win_rate = self._test_play(test_episodes=100)
                print(f'Best win rate: {self.agent.best_winrate:.1f}')
                if win_rate > self.best_win_rate:  # 不能用agent.best_winrate，它已经更新了
                    best_filename = os.path.join(
                        self.checkpoint_dir,
                        f"best_dqn_{self.env.rows}x{self.env.cols}x{self.env.mines}_ep{self.agent.episodes}_winrate{win_rate:.1f}.pth")
                    self.agent.save(best_filename)
                    print(f"Best win rate of {win_rate:.1f} saved at episode {self.agent.episodes}.")
                    self.best_win_rate = win_rate

    def _test_play(self, test_episodes):
        game_loses = []
        revealed_cnts = []
        rewards = []
        losses = []
        for _ in range(test_episodes):
            game_lose, revealed_cnt, reward, loss = self._test_episode()
            game_loses.append(game_lose)
            revealed_cnts.append(revealed_cnt)
            rewards.append(reward)
            losses.append(loss)
        win_rate = game_loses.count(False) / test_episodes
        avg_revealed_prcnt = np.average(revealed_cnts) / (self.env.rows * self.env.cols - self.env.mines) * 100
        avg_reward = np.average(rewards)
        avg_loss = np.average(losses)
        self.agent.write_test_result(win_rate, avg_revealed_prcnt, avg_reward, avg_loss)
        print(f"Test: Win rate: {win_rate:.2f}, Avg revealed%: {avg_revealed_prcnt:.2f}, Avg reward: {avg_reward:.1f}, Avg loss: {avg_loss:.5f}")
        return win_rate

    @torch.no_grad()
    def _test_episode(self):
        self.env.reset()
        state = self.env.get_normalized_state()
        rewards = []
        reveald_cnt = 0
        valid_actions = [(r, c) for r in range(self.env.rows) for c in range(self.env.cols)]
        skipped_cells = set()
        lose = False
        losses = []
        while not self.env.check_win() and not lose:
            if self.env.first_click:
                # force to make a random action if first click
                action, _ = self.agent.act(state, valid_actions, force_random=True)
            else:
                # action, _ = self.agent.act(state, valid_actions, safe_action_first=True, env=self.env)
                action, _ = self.agent.act(state, valid_actions)
            # avoid wrong action made again.
            valid_actions.remove(action)
            row, col = action
            lose, _ = self.env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=False)
            reward, done, lose, reveald_cnt, _ = self._calc_reward(row, col, lose, 0, reveald_cnt,
                                                                            skipped_cells)
            next_state = self.env.get_normalized_state()
            # self.agent.remember(state, action, reward, next_state, done)
            rewards.append(reward)
            losses.append(self.agent.calc_loss(state, action, reward, next_state, done))
            if lose:
                # game over
                break
            state = next_state
        avg_loss = np.average(losses)
        avg_reward = np.average(rewards)
        return lose, reveald_cnt, avg_reward, avg_loss


    def _train_episode(self):
        state = self.env.get_normalized_state()
        total_rewards = []
        reveald_cnt = 0
        failed_cnt = 0
        valid_actions = [(r, c) for r in range(self.env.rows) for c in range(self.env.cols)]
        skipped_cells = set()
        while not self.env.check_win() and failed_cnt <= self.MAX_RETRY:
            if self.env.first_click:
                # force to make a random action if first click
                action, _ = self.agent.act(state, valid_actions, force_random=True)
            else:
                action, _ = self.agent.act(state, valid_actions, safe_action_first=True, env=self.env)
            # avoid wrong action made again.
            valid_actions.remove(action)
            row, col = action
            lose, _ = self.env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=self.allow_retry)
            reward, done, lose, reveald_cnt, failed_cnt = self._calc_reward(row, col, lose, failed_cnt, reveald_cnt, skipped_cells)
            next_state = self.env.get_normalized_state()
            self.agent.remember(state, action, reward, next_state, done)
            total_rewards.append(reward)
            if lose:
                if self.allow_retry:
                    # don't update state, allow self.agent retry
                    # self.env.state is not updated since allow_retry=True
                    continue
                else:
                    # game over
                    break
            state = next_state

        if len(self.agent.memory) > self.batch_size and self.agent.episodes % self.train_every == 0:
            loss = self.agent.train(self.batch_size)
            if self.report_every > 0 and self.agent.episodes % self.report_every == 0:
                sys.stdout.write \
                    (f"\tAvgReward: {np.average(total_rewards):.1f} \t Loss: {loss:.5f}")
                sys.stdout.flush()
                print()
        return failed_cnt, reveald_cnt, total_rewards

    def _calc_reward(self, row, col, lose, failed_cnt, reveald_cnt, skipped_cells: set):
        if lose:
            failed_cnt += 1
            done = True
            reward = -self.agent.MAX_REWARD
            # # 周围没有翻开的格子时，惩罚更大，避免乱猜雷
            # if all(self.env.state[r][c] == CellState.UNREVEALED_EMPTY for r, c in self.env.get_around_cells(row, col)):
            #     penalty = self.agent.MAX_REWARD * 0.5
            #     reward -= penalty
        else:
            reveald_cnt += 1
            done = self.env.check_win()

            # has_any_unrevealed_empty_around_revealed_empty = False
            # row_empty, col_empty = None, None
            # for r in range(self.env.rows):
            #     for c in range(self.env.cols):
            #         if (r, c) not in skipped_cells and self.env.state[r][c] == CellState.REVEALED_EMPTY:
            #             for rr, cc in self.env.get_around_cells(r, c):
            #                 if self.env.state[rr][cc] == CellState.UNREVEALED_EMPTY:
            #                     has_any_unrevealed_empty_around_revealed_empty = True
            #                     row_empty, col_empty = rr, cc
            #                     # 注意这个break退不出外面的二层循环，下一次的会被替换掉
            #                     break
            #                 skipped_cells.add((rr, cc))

            # has_revealed_empty_around = False
            # for r, c in self.env.get_around_cells(row, col):
            #     if self.env.state[r][c] == CellState.REVEALED_EMPTY:
            #         has_revealed_empty_around = self.env.is_neighbor(row, col, r, c)

            # for r, c in self.env.get_around_cells(row, col):
            #     if self.env.state[r][c] != CellState.UNREVEALED_EMPTY:
            #         all_unrevealed_empty_around = False

            # 如果有已翻开的空白，则必须点它周围，否则相当于输
            # if has_any_unrevealed_empty_around_revealed_empty and not self.env.is_neighbor(row, col, row_empty, col_empty):
            #     reward = -self.agent.MAX_REWARD
            #     done = lose = True
            # else:
                # x点对了也不能乱翻，如果周围都是没翻开的，要严重惩罚，但不能算输。需要吗。先不写，他要学的就是这个，奖励跟周围数字数量不是正比关系。

                # # 周围有空白格子时，当前格一定不是雷，从第二步开始，点得越早奖励越大
                # special_award_prcnt = map_range(reveald_cnt, self.env.rows * self.env.cols - self.env.mines, 0,
                #                                 in_min=2,
                #                                 out_min=0.5) if has_revealed_empty_around and not self.env.first_click else 0

                # # 奖励不能比输的惩罚多，要不然会不在意输的惩罚，所以不超过总奖励的90%
                # if explore_prcnt >= 0.9:
                #     # 0.9 ~ 1
                #     total_reward_prcnt = explore_prcnt
                # else:
                #     # 0 ~ 0.9
                #     total_reward_prcnt = min(explore_prcnt + special_award_prcnt, 0.9)
                # total_reward_prcnt = 0.1
                # reward = self.agent.MAX_REWARD * total_reward_prcnt
            total_reward_prcnt = 0.1
            reward = self.agent.MAX_REWARD * total_reward_prcnt
        return reward, done, lose, reveald_cnt, failed_cnt

    def print_status(self, *args):
        failed_cnt, reveald_cnt, reward = args
        sys.stdout.write \
            (f"\nEp: {self.agent.episodes} \t Revealed: {reveald_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {self.agent.epsilon:.2f}")


if __name__ == '__main__':
    exp = ExperimentNormal()
    exp.run()

