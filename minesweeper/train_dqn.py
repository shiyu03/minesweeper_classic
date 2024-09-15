import argparse
import os
import sys
import time
from abc import abstractmethod

import numpy as np

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
        self.MAX_REPLAY = None
        self.train_every = None
        self.test_every = None
        self.setup()

    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--gui', action='store_true', help='Enable visualization of the agent\'s behavior')
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
        args = parser.parse_args()
        args.ckpt = os.path.join(args.ckpt, f'dqn_{args.rows}x{args.cols}x{args.mines}')
        return args

    @abstractmethod
    def setup(self): ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def _calc_reward(self, *args): ...

    @abstractmethod
    def _test_play(self): ...

    def print_board(self, mine_position, histories):
        # 打印棋盘状态
        row, col = mine_position
        sys.stdout.write("\033c")  # 清除终端
        if self.args.eval:
            sys.stdout.write \
                ('\n'.join(f'Ep: {history[0]} \t Revealed: {history[1]}' for history in histories))
            sys.stdout.write('\n')
        sys.stdout.write(f"Reveal: ({row}, {col}) - {self.env.state[row][col].name}\n")
        board_str = self.env.board_to_string(mine_position, flag_mines=True)
        sys.stdout.write(board_str)


class ExperimentNormal(Experiment):
    def setup(self):
        self.env = MinesweeperEnv(rows=self.args.rows, cols=self.args.cols, mines=self.args.mines)
        input_shape = (self.env.rows, self.env.cols)
        output_size = self.env.rows * self.env.cols
        self.agent = DQNAgent(input_shape, output_size, self.args.log_suffix, self.args.eval)

        self.save_every = self.args.save_every
        self.checkpoint_dir = self.args.ckpt
        os.makedirs(self.args.ckpt, exist_ok=True)
        self.batch_size = self.args.ba
        self.allow_retry = True if not self.args.eval else False
        self.MAX_RETRY = self.args.mines if self.allow_retry else 1
        self.MAX_REPLAY = 1
        self.train_every = 10
        self.test_every = self.args.test_every

        # 尝试加载检查点
        try:
            self.agent.load(self.checkpoint_dir, self.args.rows, self.args.cols, self.args.mines)
            if self.args.reseteps:
                self.agent.epsilon = 1.0
                print("WARNING: Reset epsilon to 1.0!")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        if self.args.eval:
            self.agent.model.eval()

    def run(self):
        records = {
            'failed_cnt': 0,
            'reveald_cnt': 0,
            'loss': 0,
            'reward': 0,
        }
        loss = 0
        last_loss = 0
        histories = []
        while True:
            self.env.reset()
            game_played = 0
            while game_played < self.MAX_REPLAY:
                self.agent.episodes += 1
                game_played += 1
                failed_cnt, reveald_cnt, total_rewards, last_loss = (
                    self._train_episode(histories, last_loss)
                )

                print('')
                records['failed_cnt'] += failed_cnt
                records['reveald_cnt'] += reveald_cnt
                records['loss'] += loss
                records['reward'] += np.average(total_rewards)
                self.env.replay()

                if not self.args.eval and self.agent.episodes % self.save_every == 0:
                    avg_fail = records['failed_cnt'] / self.save_every
                    # avg_revealed_percent = records['reveald_cnt'] / self.save_every / (self.env.rows * self.env.cols - self.env.mines) * 100
                    avg_reward = records['reward'] / self.save_every
                    avg_loss = records['loss'] / self.save_every
                    checkpoint_filename = os.path.join(
                        self.checkpoint_dir,
                        f"dqn_{self.env.rows}x{self.env.cols}x{self.env.mines}_ep{self.agent.episodes}_eps{self.agent.epsilon:.3f}_ls{avg_loss:.3f}_ba{self.batch_size}_reward{avg_reward:.1f}_fail{avg_fail:.1f}.pth")
                    self.agent.save(checkpoint_filename)
                    print(f"Checkpoint saved at episode {self.agent.episodes}.")
                    records['failed_cnt'] = 0
                    records['reveald_cnt'] = 0
                    records['loss'] = 0
                    records['reward'] = 0

                if self.test_every > 0 and self.agent.episodes % self.test_every == 0:
                    self._test_play()

    def _test_play(self):
        win_rate, avg_revealed_prcnt = self.agent.test_play(self.env, test_episodes=100, allow_recursive=False)
        print(f"Test: Win rate: {win_rate:.2f}, Avg revealed%: {avg_revealed_prcnt:.2f}")

    def _train_episode(self, histories, last_loss):
        state = self.env.get_normalized_state()
        total_rewards = []
        reveald_cnt = 0
        failed_cnt = 0
        valid_actions = [(r, c) for r in range(self.env.rows) for c in range(self.env.cols)]
        skipped_cells = set()
        while not self.env.check_win() and failed_cnt <= self.MAX_RETRY:
            if self.env.first_click:
                # force to make a random action if first click
                action = self.agent.act(state, valid_actions, force_random=True)
            else:
                action = self.agent.act(state, valid_actions)
            # avoid wrong action made again.
            valid_actions.remove(action)
            reward, done, lose, reveald_cnt, failed_cnt = self._calc_reward(action, failed_cnt, reveald_cnt, skipped_cells)
            next_state = self.env.get_normalized_state()
            if not self.args.eval:
                self.agent.remember(state, action, reward, next_state, done)
            if self.args.gui:
                self.print_board(action, histories)
                self.print_status(failed_cnt, reveald_cnt, reward)
            else:
                self.print_status(failed_cnt, reveald_cnt, reward)
            sys.stdout.flush()
            total_rewards.append(reward)
            if self.args.eval:
                time.sleep(0.2)
            if lose:
                if self.allow_retry:
                    # don't update state, allow self.agent retry
                    # self.env.state is not updated since allow_retry=True
                    continue
                else:
                    # game over
                    break
            state = next_state
        if self.args.eval:
            histories.append((self.agent.episodes, reveald_cnt))
            time.sleep(2)

        if not self.args.eval and len(
                self.agent.memory) > self.batch_size and self.agent.episodes % self.train_every == 0:
            loss = self.agent.train(self.batch_size)
            sys.stdout.write \
                (f"\tAvgReward: {np.average(total_rewards):.1f} \t Loss: {loss:.3f}, Δ: {loss - last_loss:.3f}")
            sys.stdout.flush()
            last_loss = loss
        return failed_cnt, reveald_cnt, total_rewards, last_loss

    def _calc_reward(self, action, failed_cnt, reveald_cnt, skipped_cells: set):
        row, col = action
        lose, _ = self.env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                     allow_retry=self.allow_retry)
        if lose:
            failed_cnt += 1
            done = True
            reward = -self.agent.MAX_REWARD
            # 周围没有翻开的格子时，惩罚更大，避免乱猜雷
            if all(self.env.state[r][c] == CellState.UNREVEALED_EMPTY for r, c in self.env.get_around_cells(row, col)):
                penalty = self.agent.MAX_REWARD * 0.5
                reward -= penalty
        else:
            reveald_cnt += 1
            done = self.env.check_win()

            has_any_unrevealed_empty_around_revealed_empty = False
            row_empty, col_empty = None, None
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    if (r, c) not in skipped_cells and self.env.state[r][c] == CellState.REVEALED_EMPTY:
                        for rr, cc in self.env.get_around_cells(r, c):
                            if self.env.state[rr][cc] == CellState.UNREVEALED_EMPTY:
                                has_any_unrevealed_empty_around_revealed_empty = True
                                row_empty, col_empty = rr, cc
                                break
                            skipped_cells.add((rr, cc))

            has_revealed_empty_around = False
            for r, c in self.env.get_around_cells(row, col):
                if self.env.state[r][c] == CellState.REVEALED_EMPTY:
                    has_revealed_empty_around = self.env.is_neighbor(row, col, r, c)

            # for r, c in self.env.get_around_cells(row, col):
            #     if self.env.state[r][c] != CellState.UNREVEALED_EMPTY:
            #         all_unrevealed_empty_around = False

            # 如果有已翻开的空白，则必须点它周围，否则相当于输
            if has_any_unrevealed_empty_around_revealed_empty and not self.env.is_neighbor(row, col, row_empty, col_empty):
                reward = -self.agent.MAX_REWARD
                done = lose = True
            else:
                # 点对了也不能乱翻，如果周围都是没翻开的，要严重惩罚，但不能算输。需要吗。先不写，他要学的就是这个，奖励跟周围数字数量不是正比关系。
                # 开的越多，奖励越多
                explore_prcnt = reveald_cnt / (self.env.rows * self.env.cols - self.env.mines)

                # 周围有空白格子时，当前格一定不是雷，从第二步开始，点得越早奖励越大
                special_award_prcnt = map_range(reveald_cnt, self.env.rows * self.env.cols - self.env.mines, 0,
                                                in_min=2,
                                                out_min=0.5) if has_revealed_empty_around and not self.env.first_click else 0

                # 奖励不能比输的惩罚多，要不然会不在意输的惩罚，所以不超过总奖励的90%
                if explore_prcnt >= 0.9:
                    # 0.9 ~ 1
                    total_reward_prcnt = explore_prcnt
                else:
                    # 0 ~ 0.9
                    total_reward_prcnt = min(explore_prcnt + special_award_prcnt, 0.9)
                reward = self.agent.MAX_REWARD * total_reward_prcnt
        # 错的次数越多，当前盘面越难，尽量避免
        # TODO 这个规则可以后期再微调，前期可以关掉
        # difficulty_penalty_prcnt = map_range(failed_cnt, self.MAX_RETRY, 0.2, in_min=0, out_min=0)
        # reward -= self.agent.MAX_REWARD * difficulty_penalty_prcnt
        return reward, done, lose, reveald_cnt, failed_cnt

    def print_status(self, *args):
        failed_cnt, reveald_cnt, reward = args
        sys.stdout.write \
            (f"\nEp: {self.agent.episodes} \t Revealed: {reveald_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {self.agent.epsilon:.2f}")


if __name__ == '__main__':
    exp = ExperimentNormal()
    exp.run()

