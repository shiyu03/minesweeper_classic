import copy
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


class DQNWorkerAgent:
    def __init__(self, id, gamma, model):
        # epsilon, lr, episodes, MAX_REWARD, epsilon_min, epsilon_decay, epsilon_decay_step,
        self.id = id
        # self.memory = memory
        self.gamma = gamma
        # self.epsilon = epsilon
        # self.epsilon_min = epsilon_min
        # self.lr = lr
        # self.lr_decay = lr_decay
        # self.lr_decay_step = lr_decay_step
        # self.lr_min = lr_min
        # self.epsilon_decay = epsilon_decay
        # self.epsilon_decay_step = epsilon_decay_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.episodes = episodes
        # self.MAX_REWARD = MAX_REWARD
        self.criterion = nn.MSELoss()
        self.model = model
        # self.optimizer = optimizer
        # self.init_state_dict_optimizer = None

    # def load_model(self, state_dict):
    #     self.model.load_state_dict(state_dict)

    # def new_or_update_optimizer(self):
    #     if self.optimizer is None:
    #         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    #         if self.init_state_dict_optimizer is not None:
    #             self.optimizer.load_state_dict(self.init_state_dict_optimizer)
    #         return
    #     last_state_dict_optimizer = self.optimizer.state_dict()
    #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    #     self.optimizer.load_state_dict(last_state_dict_optimizer)
    #
    # def update_optimizer_from(self, state_dict_optimizer):
    #     self.init_state_dict_optimizer = state_dict_optimizer
    #
    # def state_dict_optimizer(self):
    #     if self.optimizer is None:
    #         return self.init_state_dict_optimizer
    #     return self.optimizer.state_dict()

    # 传参数版
    # def train(self, memory, batch_size):
    #     minibatch = random.sample(memory, batch_size)
    #     avg_loss = 0
    #     avg_reward = 0
    #     for state, action, reward, next_state, done in minibatch:
    #         avg_reward += reward
    #         self.optimizer.zero_grad()
    #         state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
    #         next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
    #         reward = torch.FloatTensor([reward]).to(self.device)
    #         target = reward
    #         if not done:
    #             target = reward + self.gamma * torch.max(self.model(next_state)[0])
    #         target_f = self.model(state)
    #         target_f[0][self._action_to_index(action, state.shape[3])] = target
    #         loss = self.criterion(self.model(state), target_f)
    #         avg_loss += loss.item()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #     if self.epsilon > self.epsilon_min and self.episodes % self.epsilon_decay_step == 0:
    #         self.epsilon *= self.epsilon_decay
    #
    #     # if self.lr > self.lr_min and self.episodes % self.lr_decay_step == 0:
    #     #     self.lr *= self.lr_decay
    #
    #     avg_loss /= batch_size
    #     avg_reward /= batch_size
    #     return avg_loss, avg_reward, self.epsilon, self.optimizer.state_dict()['param_groups'][0]['lr']

    # TODO 传梯度版 只算一个epoch
    def train(self, state, action, reward, next_state, done):
        self.model.zero_grad()
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state)[0])
        target_f = self.model(state)
        target_f[0][self._action_to_index(action, state.shape[3])] = target
        loss = self.criterion(self.model(state), target_f)
        loss.backward()
        grads = {name: param.grad.detach() for name, param in self.model.named_parameters()}
        return grads, loss.item()
        #
        # if self.epsilon > self.epsilon_min and self.episodes % self.epsilon_decay_step == 0:
        #     self.epsilon *= self.epsilon_decay

        # if self.lr > self.lr_min and self.episodes % self.lr_decay_step == 0:
        #     self.lr *= self.lr_decay

        # avg_loss /= batch_size
        # avg_reward /= batch_size

    def _action_to_index(self, action, cols):
        return action[0] * cols + action[1]

# TODO: 后期降低batch size
# TODO 调整 epsilon/lr 初始值/decay/decay step, setup中的xxx_every等
class FedDQNAgent:
    def __init__(self, *, comment='', model=None, eval=False):
        self.gamma = 0.99    # discount rate 0.9 0.95 0.99 越大越重视未来奖励
        self.criterion = nn.MSELoss()
        self.MAX_REWARD = 1
        self.eval = eval
        self.writer = SummaryWriter(comment=comment) if not eval else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_decay_step = 1
        # self.lr_min = 1e-5
        # self.lr_decay = 0.99
        # self.lr_decay_step = 100
        self.memory_size = 40000

        # loadable parameters
        if model is not None:
            self.model = model
        else:
            self.model = DQN().to(self.device)
            self.model.share_memory()
        self.lr = 0.001  # 0.001, 0.0005, and 0.0001
        self.epsilon = 1.0  # exploration rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) if not eval else None
        self.episodes = 0
        self.memory = deque(maxlen=self.memory_size) if not eval else None
        self.best_winrate = 0.0

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'episodes': self.episodes,
            'memory': list(self.memory),
            'best_winrate': self.best_winrate,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)

    def load(self, checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"dqn_") and f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint found in the directory.")

        # Extract episode numbers and find the file with the largest episode number
        episode_numbers = [int(re.search(r"ep(\d+)_", f).group(1)) for f in checkpoint_files]
        max_episode_file = checkpoint_files[episode_numbers.index(max(episode_numbers))]

        checkpoint_file = os.path.join(checkpoint_dir, max_episode_file)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.share_memory()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # note: saved lr is not used
        self.lr = checkpoint['lr']
        self.episodes = checkpoint['episodes']
        self.memory = deque(checkpoint['memory'], maxlen=self.memory_size)
        self.best_winrate = checkpoint['best_winrate']
        self.model.train()
        print(f"Checkpoint loaded from {checkpoint_file}")

    def load_best(self, checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"best") and f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint found in the directory.")
        win_rates = [float(re.search(r"winrate([\d.]+)\.pth", f).group(1)) for f in checkpoint_files]
        max_episode_file = checkpoint_files[win_rates.index(max(win_rates))]
        checkpoint_file = os.path.join(checkpoint_dir, max_episode_file)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.epsilon = checkpoint['epsilon']
        self.lr = checkpoint['lr']
        self.episodes = checkpoint['episodes']
        self.memory = deque(checkpoint['memory'], maxlen=self.memory_size)
        self.best_winrate = checkpoint.get('best_winrate', 0.0)
        self.model.train()
        print(f"Checkpoint loaded from {checkpoint_file}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, force_random=False, safe_action_first=False, env=None):
        if safe_action_first:
            for r, c in valid_actions:
                for rr, cc in env.get_around_cells(r, c):
                    if env.state[rr][cc] == CellState.REVEALED_EMPTY:
                        safe_action = (r, c)
                        return safe_action, None
        if force_random or not self.eval and np.random.rand() <= self.epsilon:
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

    def global_train_episode(self, grads_glob, avg_loss):
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = grads_glob[name]
        self.optimizer.step()

        if self.epsilon > self.epsilon_min and self.episodes % self.epsilon_decay_step == 0:
            self.epsilon *= self.epsilon_decay

        # Log metrics to TensorBoard
        self.writer.add_scalar('Train: Loss', avg_loss, self.episodes)
        self.writer.add_scalar('Train: Epsilon', self.epsilon, self.episodes)
        # self.writer.add_scalar('Train: Learning Rate', lr, self.episodes)
        # self.writer.add_scalar('Train: Avg Reward', avg_reward, self.episodes)

    # def update_train_episode(self, w_glob, avg_loss, avg_reward, epsilon, lr):
    #     self.model.load_state_dict(w_glob)
    #
    #     self.epsilon = epsilon
    #     # self.lr = lr
    #
    #     # Log metrics to TensorBoard
    #     self.writer.add_scalar('Train: Loss', avg_loss, self.episodes)
    #     self.writer.add_scalar('Train: Epsilon', self.epsilon, self.episodes)
    #     self.writer.add_scalar('Train: Learning Rate', lr, self.episodes)
    #     self.writer.add_scalar('Train: Avg Reward', avg_reward, self.episodes)

    def write_test_result(self, win_rate, revealed_percent, reward, loss):
        self.best_winrate = max(self.best_winrate, win_rate)
        if not self.eval:
            self.writer.add_scalar('Test: Win rate', win_rate, self.episodes)
            self.writer.add_scalar('Test: Avg revealed percent', revealed_percent, self.episodes)
            self.writer.add_scalar('Test: Avg Reward', reward, self.episodes)
            self.writer.add_scalar('Test: Loss', loss, self.episodes)

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
        return loss


def FedAvg(ws, device):
    for w in ws:
        for k in w.keys():
            w[k] = w[k].to(device)
    w_avg = copy.deepcopy(ws[0])
    for k in w_avg.keys():
        for i in range(1, len(ws)):
            w_avg[k] += ws[i][k]
        w_avg[k] = torch.div(w_avg[k], len(ws))
    return w_avg


def GradAvg(grads_list):
    avg_grads = {}
    for grads in grads_list:
        for name, grad in grads.items():
            if name not in avg_grads:
                avg_grads[name] = grad.clone()
            else:
                avg_grads[name] += grad
    for name in avg_grads:
        avg_grads[name] = torch.div(avg_grads[name], len(grads_list))
    return avg_grads