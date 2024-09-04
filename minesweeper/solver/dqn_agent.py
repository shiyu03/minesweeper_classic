import argparse
import os
import random
import re
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from minesweeper.game.minesweeper_env import MinesweeperEnv, CellState
from minesweeper.const import ROOT

class DQN(nn.Module):
    def __init__(self, input_shape, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 128)
        # TODO fine-tune on different output_size
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate 0.9 0.95 0.99 越大越重视未来奖励
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001  # 0.001, 0.0005, and 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_shape, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.episodes = 0
        self.writer = SummaryWriter()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, force_random=False):
        if np.random.rand() <= self.epsilon or force_random:
            return random.choice(valid_actions)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return valid_actions[np.argmax([act_values[0][self._action_to_index(a, state.shape[3])].item() for a in valid_actions])]

    def _action_to_index(self, action, cols):
        return action[0] * cols + action[1]

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        avg_loss = 0
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0])
            target_f = self.model(state)
            target_f[0][self._action_to_index(action, state.shape[3])] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            avg_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        avg_loss /= batch_size
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss', avg_loss, self.episodes)
        self.writer.add_scalar('Epsilon', self.epsilon, self.episodes)
        return avg_loss

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'memory': list(self.memory)
        }, filename)

    def load(self, checkpoint_dir, rows, cols, mines):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"dqn_{rows}x{cols}x{mines}") and f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint found in the directory.")

        # Extract episode numbers and find the file with the largest episode number
        episode_numbers = [int(re.search(r"ep(\d+)_", f).group(1)) for f in checkpoint_files]
        max_episode_file = checkpoint_files[episode_numbers.index(max(episode_numbers))]

        checkpoint_file = os.path.join(checkpoint_dir, max_episode_file)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episodes = checkpoint['episodes']
        self.memory = deque(checkpoint['memory'], maxlen=2000)
        self.model.train()
        print(f"Checkpoint loaded from {checkpoint_file}")

def experiment(rows, cols, mines, agent, episodes, batch_size):
    env = MinesweeperEnv(rows, cols, mines)
    input_shape = (env.rows, env.cols)
    output_size = env.rows * env.cols

def map_range(value, in_max, out_max, in_min=0, out_min=0):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', action='store_true', help='Enable visualization of the agent\'s behavior')
    parser.add_argument('-r', '--rows', default=8, type=int, help='Number of rows in the Minesweeper grid')
    parser.add_argument('-c', '--cols', default=8, type=int, help='Number of columns in the Minesweeper grid')
    parser.add_argument('-m', '--mines', default=10, type=int, help='Number of mines in the Minesweeper grid')
    parser.add_argument('-e', '--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--ckpt', default='checkpoint', help='Checkpoint path')
    parser.add_argument('--reseteps', action='store_true', help='reset epsilon')
    parser.add_argument('--ba', default=64, type=int, help='Batch size')
    parser.add_argument('--save_every', default=1000, type=int, help='Save checkpoint every n episodes')
    args = parser.parse_args()

    env = MinesweeperEnv(rows=args.rows, cols=args.cols, mines=args.mines)
    input_shape = (env.rows, env.cols)
    output_size = env.rows * env.cols
    agent = DQNAgent(input_shape, output_size)
    save_every = args.save_every
    checkpoint_dir = os.path.join(ROOT, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_size = args.ba
    allow_retry = True if not args.eval else False
    MAX_RETRY = args.mines if allow_retry else 1
    allow_replay = True
    MAX_REPLAY = 5
    train_every = 10

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
            state = env.get_normalized_state()
            agent.episodes += 1
            game_played += 1
            total_reward = []
            reveald_cnt = 0
            failed_cnt = 0
            valid_actions = [(r, c) for r in range(env.rows) for c in range(env.cols)]
            while not env.check_win() and failed_cnt <= MAX_RETRY:
                if env.first_click:
                    # force to make a random action if first click
                    action = agent.act(state, valid_actions, force_random=True)
                else:
                    action = agent.act(state, valid_actions)
                # avoid wrong action made again.
                valid_actions.remove(action)
                row, col = action
                lose = env.make_move(row, col, allow_click_revealed_num=False, allow_recursive=False, allow_retry=allow_retry)
                # TODO maybe add flagging?
                # TODO maybe try all possible actions?
                if lose:
                    failed_cnt += 1
                    done = True
                    reward = -100
                    # 周围没有翻开的格子时，惩罚更大，避免乱猜雷
                    if all(env.state[r][c] == CellState.UNREVEALED_EMPTY for r, c in env.get_around_cells(row, col)):
                        reward -= 100
                else:
                    reveald_cnt += 1
                    done = env.check_win()
                    base_reward = 10
                    has_empty_around = any(env.state[r][c] == CellState.REVEALED_EMPTY for r, c in env.get_around_cells(row, col))
                    reveal_percent = reveald_cnt / (env.rows * env.cols - env.mines)
                    # 周围有空白格子时，当前格一定不是雷，点得越早越好
                    special_award = map_range(reveald_cnt, env.rows * env.cols - env.mines, 0, in_min=2, out_min=500) if has_empty_around else 0
                    # 开的越多，奖励越多
                    reveal_percent_award = reveal_percent * 100
                    reward = base_reward + reveal_percent_award + special_award
                # 错的次数越多，当前盘面越难，尽量避免
                difficulty_penalty = map_range(failed_cnt, MAX_RETRY, 50, in_min=0, out_min=0)
                reward -= difficulty_penalty
                next_state = env.get_normalized_state()
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
                    sys.stdout.write(f"Ep: {agent.episodes} \t Revealed: {reveald_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {agent.epsilon:.4f}\n")
                else:
                    sys.stdout.write(f"\nEp: {agent.episodes} \t Revealed: {reveald_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {agent.epsilon:.2f}")
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
            records['loss'] += loss
            records['reward'] += np.average(total_reward)
            total_reward = []
            env.replay()

            if not args.eval and agent.episodes % save_every == 0:
                avg_fail = records['failed_cnt'] / save_every
                # avg_revealed_percent = records['reveald_cnt'] / save_every / (env.rows * env.cols - env.mines) * 100
                avg_reward = records['reward'] / save_every
                avg_loss = records['loss'] / save_every
                checkpoint_filename = os.path.join(checkpoint_dir, f"dqn_{env.rows}x{env.cols}x{env.mines}_ep{agent.episodes}_eps{agent.epsilon:.3f}_ba{batch_size}_reward{avg_reward:.1f}_fail{avg_fail:.1f}_ls{avg_loss:.3f}.pth")
                agent.save(checkpoint_filename)
                print(f"Checkpoint saved at episode {agent.episodes}.")
                records['failed_cnt'] = 0
                records['reveald_cnt'] = 0
                records['loss'] = 0
                records['reward'] = 0
