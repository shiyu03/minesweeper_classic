import argparse
import os
import sys
import time

import numpy as np
import torch

from solver.dqn_agent import map_range, DQNAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from minesweeper.game.minesweeper_env import MinesweeperEnv, CellState
from minesweeper.const import ROOT


class DQNAgentWFlag(DQNAgent):
    def _action_to_index(self, action, cols):
        row, col, flag = action
        return (row * cols + col) * 2 + flag

    @torch.no_grad()
    def test_play(self, env, test_episodes=100):
        result = []
        total_revealed = []
        total_flagged = []
        for ep in range(test_episodes):
            env.reset()
            reveald_cnt = 0
            correct_flagged_cnt = 0
            lose = False
            valid_actions_reveal = [(r, c, 0) for r in range(env.rows) for c in range(env.cols)]
            valid_actions_flag = [(r, c, 1) for r in range(env.rows) for c in range(env.cols)]
            while not env.check_win() and not lose:
                state = env.get_normalized_state()
                if env.first_click:
                    action = self.act(state, valid_actions_reveal, force_random=True)
                else:
                    action = self.act(state, valid_actions_reveal + valid_actions_flag)
                row, col, flag = action
                if flag == 1:
                    valid_actions_flag.remove((row, col, 1))
                    env.make_move(row, col, flag=True, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=False)
                    if env.board[row][col] == -1:
                        correct_flagged_cnt += 1
                        try:
                            valid_actions_reveal.remove((row, col, 0))
                        except ValueError:
                            # 上一把试错踩雷已经去掉了这个格子
                            pass
                    else:
                        lose = True
                else:
                    lose = env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=False)
                    valid_actions_reveal.remove((row, col, 0))

                    try:
                        valid_actions_flag.remove((row, col, 1))
                    except ValueError:
                        # 上一把flag错了已经去掉了这个格子
                        pass
                    reveald_cnt += 1

            result.append(lose)
            total_revealed.append(reveald_cnt)
            total_flagged.append(correct_flagged_cnt)

        win_rate = result.count(False) / test_episodes
        avg_revealed = np.average(total_revealed)
        avg_flag = np.average(total_flagged)
        self.writer.add_scalar('Win rate', win_rate, self.episodes)
        self.writer.add_scalar('Avg revealed', avg_revealed, self.episodes)
        self.writer.add_scalar('Avg correct flag', avg_flag, self.episodes)
        return win_rate, avg_revealed, avg_flag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', action='store_true', help='Enable visualization of the agent\'s behavior')
    parser.add_argument('-r', '--rows', default=8, type=int, help='Number of rows in the Minesweeper grid')
    parser.add_argument('-c', '--cols', default=8, type=int, help='Number of columns in the Minesweeper grid')
    parser.add_argument('-m', '--mines', default=10, type=int, help='Number of mines in the Minesweeper grid')
    parser.add_argument('-e', '--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--ckpt', default=os.path.join(ROOT, "checkpoint_wflag"), help='Checkpoint path')
    parser.add_argument('--log_suffix', default="normal", help='suffix for log dirname')
    parser.add_argument('--reseteps', action='store_true', help='reset epsilon')
    parser.add_argument('--ba', default=64, type=int, help='Batch size')
    parser.add_argument('--save_every', default=5000, type=int, help='Save checkpoint every n episodes')
    parser.add_argument('--test_every', default=1000, type=int, help='Test play every n episodes')
    args = parser.parse_args()

    env = MinesweeperEnv(rows=args.rows, cols=args.cols, mines=args.mines)
    input_shape = (env.rows, env.cols)
    output_size = env.rows * env.cols * 2  # may be click or flag
    agent = DQNAgentWFlag(input_shape, output_size, args.log_suffix)
    save_every = args.save_every
    checkpoint_dir = args.ckpt
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_size = args.ba
    allow_retry = True if not args.eval else False
    MAX_RETRY = args.mines if allow_retry else 1
    MAX_REPLAY = 1
    train_every = 10
    test_every = args.test_every

    # 尝试加载检查点
    try:
        agent.load(checkpoint_dir, args.rows, args.cols, args.mines)
        if args.reseteps:
            agent.epsilon = 1.0
            print("WARNING: Reset epsilon to 1.0!")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    if args.eval:
        agent.model.eval()
    
    records = {
        'failed_cnt': 0,
        'reveald_cnt': 0,
        'flagged_cnt': 0,
        'loss': 0,
        'reward': 0,
    }

    loss = 0
    last_loss = 0
    histories = []
    while True:
        env.reset()
        game_played = 0
        while game_played < MAX_REPLAY:
            state = env.get_normalized_state_wflag()
            agent.episodes += 1
            game_played += 1
            total_reward = []
            reveald_cnt = 0
            failed_cnt = 0
            flagged_cnt = 0
            valid_actions_reveal = [(r, c, 0) for r in range(env.rows) for c in range(env.cols)]
            valid_actions_flag = [(r, c, 1) for r in range(env.rows) for c in range(env.cols)]
            while not env.check_win() and failed_cnt <= MAX_RETRY:
                if env.first_click:
                    # force to make a random action if first click
                    action = agent.act(state, valid_actions_reveal, force_random=True)
                else:
                    action = agent.act(state, valid_actions_reveal + valid_actions_flag)
                row, col, flag = action
                # avoid wrong action made again.
                if flag == 1:
                    valid_actions_flag.remove((row, col, 1))
                    lose = env.make_move(row, col, flag=True, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=allow_retry)
                    has_num_around = any(env.is_neighbor(row, col, r, c) for r, c in env.get_around_cells(row, col))
                    if lose or has_num_around:
                        # 如果flag错了直接结束，防止影响后续策略
                        # flag对了也不能乱打，否则也算输
                        done = True
                        reward = -agent.MAX_REWARD
                    else:
                        # 但是flag对了比开雷的奖励还要大，防止agent不flag
                        flagged_cnt += 1
                        done = False
                        reward = agent.MAX_REWARD
                        try:
                            valid_actions_reveal.remove((row, col, 0))
                        except ValueError:
                            # 上一把试错踩雷已经去掉了这个格子
                            pass

                else:
                    lose = env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=allow_retry)
                    valid_actions_reveal.remove((row, col, 0))
                    if lose:
                        failed_cnt += 1
                        done = True
                        reward = -agent.MAX_REWARD
                        # 周围没有翻开的格子时，惩罚更大，避免乱猜雷
                        if all(env.state[r][c] == CellState.UNREVEALED_EMPTY for r, c in env.get_around_cells(row, col)):
                            penalty = agent.MAX_REWARD * 0.5
                            reward -= penalty
                    else:
                        try:
                            valid_actions_flag.remove((row, col, 1))
                        except ValueError:
                            # 上一把flag错了已经去掉了这个格子
                            pass
                        reveald_cnt += 1
                        done = env.check_win()

                        has_empty = False
                        row_empty, col_empty = 0, 0
                        for r in range(env.rows):
                            for c in range(env.cols):
                                if env.state[r][c] == CellState.UNREVEALED_EMPTY:
                                    has_empty = True
                                    row_empty = r
                                    col_empty = c
                                    break
                        has_empty_around = env.is_neighbor(row, col, row_empty, col_empty)
                        if has_empty and not has_empty_around:
                            # 如果有空白格子不点反而要惩罚，相当于输
                            reward = -agent.MAX_REWARD
                        else:
                            # 开的越多，奖励越多
                            explore_pcnt = (reveald_cnt + flagged_cnt) / (env.rows * env.cols)

                            # 周围有空白格子时，当前格一定不是雷，从第二步开始，点得越早奖励越大
                            special_award_pcnt = map_range(reveald_cnt, env.rows * env.cols - env.mines, 0, in_min=2, out_min=0.5) if has_empty_around and not env.first_click else 0

                            # 奖励不能比输的惩罚多，要不然会不在意输的惩罚，所以不超过总奖励的90%
                            if explore_pcnt >= 0.9:
                                # 0.9 ~ 1
                                total_reward_pcnt = explore_pcnt
                            else:
                                # 0 ~ 0.9
                                total_reward_pcnt = min(explore_pcnt + special_award_pcnt, 0.9)
                            reward = agent.MAX_REWARD * total_reward_pcnt

                    # 错的次数越多，当前盘面越难，尽量避免
                    difficulty_penalty_pcnt = map_range(failed_cnt, MAX_RETRY, 0.2, in_min=0, out_min=0)
                    reward -= agent.MAX_REWARD * difficulty_penalty_pcnt
                next_state = env.get_normalized_state_wflag()
                if not args.eval:
                    agent.remember(state, action, reward, next_state, done)
                if args.gui:
                    # 打印棋盘状态
                    sys.stdout.write("\033c")  # 清除终端
                    if args.eval:
                        sys.stdout.write('\n'.join(f'Ep: {history[0]} \t Revealed: {history[1]}' for history in histories))
                        sys.stdout.write('\n')
                    sys.stdout.write(f"Reveal: ({row}, {col}) - {env.state[row][col].name}\n")
                    board_str = env.board_to_string(mine_position=(row, col), flag_mines=True)
                    sys.stdout.write(board_str)
                    sys.stdout.write(f"Ep: {agent.episodes} \t Revealed: {reveald_cnt} Flagged: {flagged_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {agent.epsilon:.4f}\n")
                else:
                    sys.stdout.write(f"\nEp: {agent.episodes} \t Revealed: {reveald_cnt} Flagged: {flagged_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {agent.epsilon:.2f}")
                sys.stdout.flush()
                total_reward.append(reward)
                if args.eval:
                    time.sleep(0.2)
                if lose:
                    if allow_retry:
                        # don't update state, allow agent retry
                        # env.state is not updated since allow_retry=True
                        continue
                    else:
                        # game over
                        break
                state = next_state
            if args.eval:
                histories.append((agent.episodes, reveald_cnt))
                time.sleep(2)
            if not args.eval and len(agent.memory) > batch_size and agent.episodes % train_every == 0:
                loss = agent.train(batch_size)
                sys.stdout.write(f"\tAvgReward: {np.average(total_reward):.1f} \t Loss: {loss:.3f}, Δ: {loss - last_loss:.3f}")
                sys.stdout.flush()
                last_loss = loss
            print('')
            records['failed_cnt'] += failed_cnt
            records['reveald_cnt'] += reveald_cnt
            records['flagged_cnt'] += flagged_cnt
            records['loss'] += loss
            records['reward'] += np.average(total_reward)
            total_reward = []
            env.replay()

            if not args.eval and agent.episodes % save_every == 0:
                avg_fail = records['failed_cnt'] / save_every
                # avg_revealed_percent = records['reveald_cnt'] / save_every / (env.rows * env.cols - env.mines) * 100
                avg_reward = records['reward'] / save_every
                avg_flag_percent = records['flagged_cnt'] / save_every / env.mines * 100
                avg_loss = records['loss'] / save_every
                checkpoint_filename = os.path.join(checkpoint_dir, f"dqn_{env.rows}x{env.cols}x{env.mines}_ep{agent.episodes}_eps{agent.epsilon:.3f}_ls{avg_loss:.3f}_ba{batch_size}_reward{avg_reward:.1f}_fail{avg_fail:.1f}_flag{avg_flag_percent:.1f}.pth")
                agent.save(checkpoint_filename)
                print(f"Checkpoint saved at episode {agent.episodes}.")
                records['failed_cnt'] = 0
                records['reveald_cnt'] = 0
                records['flagged_cnt'] = 0
                records['loss'] = 0
                records['reward'] = 0

            if test_every > 0 and agent.episodes % test_every == 0:
                win_rate, avg_revealed, avg_flag = agent.test_play(env, test_episodes=100)
                print(f"Test: Win rate: {win_rate:.2f}, Avg revealed: {avg_revealed:.2f}, Avg correct flag: {avg_flag:.2f}")