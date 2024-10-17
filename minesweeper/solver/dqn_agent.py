import os
import random
import re
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from game.minesweeper_env import CellState
from solver.dqn import DQN
from solver.utils import update_checkpoint

# TODO: 后期降低batch size
class DQNAgent:
    def __init__(self, comment='', eval=False):
        self.gamma = 0.99    # discount rate 0.9 0.95 0.99 越大越重视未来奖励
        self.criterion = nn.MSELoss()
        self.MAX_REWARD = 1
        self.eval = eval
        self.writer = SummaryWriter(comment=comment) if not eval else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_decay_step = 1
        self.learning_rate_min = 1e-6

        # loadable parameters
        self.model = DQN().to(self.device)
        self.learning_rate = 0.01  # 0.001, 0.0005, and 0.0001
        self.epsilon = 1.0  # exploration rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.episodes = 0
        self.memory = deque(maxlen=2000)
        self.best_winrate = 0.0
        # TODO add last epoch in load
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.98)  # adjust the learning rate during training, decay by `gamma` every `step_size` # 用了adam就不用了吧

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, force_random=False, safe_action_first=False, env=None):
        if safe_action_first:
            for r, c in valid_actions:
                for rr, cc in env.get_around_cells(r, c):
                    if env.state[rr][cc] == CellState.REVEALED_EMPTY:
                        safe_action = (r, c)
                        return safe_action, None
        if np.random.rand() <= self.epsilon or force_random:
            return random.choice(valid_actions), None
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        q_values = np.squeeze(q_values.cpu())
        mask = np.ones_like(q_values)
        for r, c in valid_actions:
            # mask == 1 表示不要的，mask == 0 表示要的
            mask[self._action_to_index((r, c), state.shape[3])] = 0
        masked_q_values = np.ma.masked_array(q_values, mask=mask)  # masked array of size (rows, cols), used for rendering qvalues in gui
        action: tuple[int, int] = self._index_to_action(np.argmax(masked_q_values), state.shape[3])
        return action, masked_q_values

    def _action_to_index(self, action, cols):
        return action[0] * cols + action[1]

    def _index_to_action(self, index, cols):
        return index // cols, index % cols

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        avg_loss = 0
        avg_reward = 0
        for state, action, reward, next_state, done in minibatch:
            avg_reward += reward
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
        avg_reward /= batch_size
        if self.epsilon > self.epsilon_min and self.episodes % self.epsilon_decay_step == 0:
            self.epsilon *= self.epsilon_decay

        if self.scheduler.get_last_lr()[-1] > self.learning_rate_min:
            self.scheduler.step()

        # Log metrics to TensorBoard
        self.writer.add_scalar('Train: Loss', avg_loss, self.episodes)
        self.writer.add_scalar('Train: Epsilon', self.epsilon, self.episodes)
        self.writer.add_scalar('Train: Learning Rate', self.scheduler.get_last_lr()[-1], self.episodes)
        self.writer.add_scalar('Train: Avg Reward', avg_reward, self.episodes)
        return avg_loss

    @torch.no_grad()
    def calc_loss(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state)[0])
        target_f = self.model(state)
        target_f[0][self._action_to_index(action, state.shape[3])] = target
        loss = self.criterion(self.model(state), target_f)
        return loss.item()

    def write_test_result(self, win_rate, revealed_percent, reward, loss):
        self.best_winrate = max(self.best_winrate, win_rate)
        if not self.eval:
            self.writer.add_scalar('Test: Win rate', win_rate, self.episodes)
            self.writer.add_scalar('Test: Avg revealed percent', revealed_percent, self.episodes)
            self.writer.add_scalar('Test: Avg Reward', reward, self.episodes)
            self.writer.add_scalar('Test: Loss', loss, self.episodes)

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'memory': list(self.memory),
            'best_winrate': self.best_winrate,
        }, filename)

    def load_best(self, checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"best") and f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint found in the directory.")
        win_rates = [float(re.search(r"winrate([\d.]+)\.pth", f).group(1)) for f in checkpoint_files]
        max_episode_file = checkpoint_files[win_rates.index(max(win_rates))]
        checkpoint_file = os.path.join(checkpoint_dir, max_episode_file)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episodes = checkpoint['episodes']
        self.memory = deque(checkpoint['memory'], maxlen=2000)
        self.best_winrate = checkpoint.get('best_winrate', 0.0)
        self.model.train()
        print(f"Checkpoint loaded from {checkpoint_file}")

    def load(self, checkpoint_dir, update_model_structure=False, freeze_old_layers=False):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"dqn_") and f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint found in the directory.")

        # Extract episode numbers and find the file with the largest episode number
        episode_numbers = [int(re.search(r"ep(\d+)_", f).group(1)) for f in checkpoint_files]
        max_episode_file = checkpoint_files[episode_numbers.index(max(episode_numbers))]

        checkpoint_file = os.path.join(checkpoint_dir, max_episode_file)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        if update_model_structure:
            mapping_model = {
                'conv1': 'conv_start',
                'conv2': 'conv_middle2',
                'conv3': 'conv_middle3',
                'conv4': 'conv_middle4',
                'conv5': 'conv_middle5',
                'conv6': 'conv_middle6',
                'conv7': 'conv_middle7',
                'conv8': 'conv_end',
            }
            checkpoint = update_checkpoint(checkpoint, mapping_model=mapping_model)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if freeze_old_layers:
            freezed_layers = [self.model.conv_start, self.model.conv_middle2, self.model.conv_middle3, self.model.conv_middle4, self.model.conv_middle5, self.model.conv_middle6, self.model.conv_middle7, self.model.conv_end]
            for layer in freezed_layers:
                print(f"Freezing layer: {layer}")
                for param in layer.parameters():
                    param.requires_grad = False
            train_layers = [self.model.conv_middle8, self.model.conv_middle9, self.model.conv_middle10, self.model.conv_middle11, self.model.conv_middle12, self.model.conv_middle13, self.model.conv_middle14, self.model.conv_middle15]
            train_params = []
            for layer in train_layers:
                print(f"Training layer: {layer}")
                for param in layer.parameters():
                    param.requires_grad = True
                    train_params.append(param)
            self.optimizer = optim.Adam(train_params, lr=self.learning_rate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if checkpoint['optimizer_state_dict']['param_groups'][0]['params'] == list(range(32)):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("Not loading optimizer state dict")
            # TODO 训练整个模型时启动，此时已经存了新模型的所有参数，不需要再动model的state_dict了 不知道这样靠不靠谱，不如直接清空整个optimizer的参数吧
            # # 旧8层模型id -> 新16层所有层模型id：
            #     # 0 ~ 13 -> 0 ~ 13
            #     # 14 ~ 15 - > 30 ~ 31
            # mapping_optimizer_old = {k: v for k, v in enumerate(list(range(0, 13+1)))}
            # mapping_optimizer_old.update({14: 30, 15: 31})
            # # 新16层中间层id -> 新16层所有层id
            #     # 0 ~ 15 -> 14 ~ 29
            # # key: 部分训练时的中间层0~15表示，value: 对应到整个模型的序号
            # mapping_optimizer_new = {k: v for k, v in enumerate(list(range(14, 29+1)))}
            # checkpoint_file_old_model = torch.load(os.path.join(checkpoint_dir, 'dqn_6x6x5_ep5330000_eps0.010_ls0.001_ba64_reward-0.0_fail0.7.pth'), map_location=self.device, weights_only=False)
            # checkpoint = update_checkpoint(checkpoint, mapping_optimizer_old=mapping_optimizer_old, mapping_optimizer_new=mapping_optimizer_new, checkpoint_file_old_model=checkpoint_file_old_model)
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epsilon = checkpoint['epsilon']
        self.episodes = checkpoint['episodes']
        self.memory = deque(checkpoint['memory'], maxlen=2000)
        self.best_winrate = checkpoint.get('best_winrate', 0.0)
        self.model.train()
        print(f"Checkpoint loaded from {checkpoint_file}")
