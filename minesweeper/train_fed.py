import os
import random
import sys
from collections import defaultdict
from multiprocessing.pool import worker

import numpy as np
import torch
import torch.multiprocessing as mp

from solver.dqn import DQN
from solver.dqn_agent_fed import FedDQNAgent, DQNWorkerAgent, FedAvg, GradAvg

from train import Experiment

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from minesweeper.game.minesweeper_env import MinesweeperEnv


class ExperimentFed(Experiment):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _parse_args():
        args = Experiment._parse_args()
        args.num_processes = 16
        # args.num_processes = 1
        # torch.set_num_threads(4)
        mp.set_start_method('spawn', force=True)
        return args

    def setup(self):
        self.agent = FedDQNAgent(comment=self.args.log_suffix, eval=self.args.eval)
        self.env = MinesweeperEnv(rows=self.args.rows, cols=self.args.cols, mines=self.args.mines)
        self.save_every = self.args.save_every
        self.checkpoint_dir = self.args.ckpt
        os.makedirs(self.args.ckpt, exist_ok=True)
        self.batch_size = self.args.ba
        self.allow_retry = True if not self.args.eval else False
        self.MAX_RETRY = self.args.mines if self.allow_retry else 1
        self.train_every = 1  # 10 次能攒350个memory 太短的话可能一直练重复的记录，容易过拟合，但是如果选的不好的话触发不了every。解法就是memory大一点，每次就不容易选中重复的
        self.test_every = self.args.test_every
        self.best_win_rate = 0.0
        # self.local_train_episodes = 10  # 每个worker训练的episode数，58左右，目标是每个人的每个batch都尽量不重复
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_local = {}
        for i in range(self.args.num_processes):
            self.models_local[i] = DQN().to(self.device)
            self.models_local[i].share_memory()

        # 尝试加载检查点
        try:
            self.agent.load(self.checkpoint_dir)
            # [worker_agent.update_optimizer_from(state_dict_optimizer) for worker_agent, state_dict_optimizer in zip(self.worker_agents, state_dict_optimizers)]
            self.best_win_rate = self.agent.best_winrate
            if self.args.reseteps:
                self.agent.epsilon = 1.0
                print("WARNING: Reset epsilon to 1.0!")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        if self.args.eval:
            self.agent.model.eval()

    def run(self):
        env = MinesweeperEnv(rows=self.args.rows, cols=self.args.cols, mines=self.args.mines)
        while True:
            self.agent.episodes += 1  # += self.local_train_episodes
            self.make_memory(env)

            if len(self.agent.memory) > self.batch_size and self.agent.episodes % self.train_every == 0:
                self._train_episode()

            if self.agent.episodes % self.save_every == 0:
                checkpoint_filename = os.path.join(
                    self.checkpoint_dir,
                    f"dqn_{self.args.rows}x{self.args.cols}x{self.args.mines}_ep{self.agent.episodes}_ba{self.batch_size}.pth")
                # state_dict_optimizers = [worker.state_dict_optimizer() for worker in self.worker_agents]
                self.agent.save(checkpoint_filename)
                print(f"Checkpoint saved at episode {self.agent.episodes}.")

            if self.test_every > 0 and self.agent.episodes % self.test_every == 0:
                win_rate = self._test_play(test_episodes=100)
                print(f'Best win rate: {self.agent.best_winrate:.1f}')
                if win_rate > self.best_win_rate:  # 不能用agent.best_winrate，它已经更新了
                    best_filename = os.path.join(
                        self.checkpoint_dir,
                        f"best_dqn_{self.args.rows}x{self.args.cols}x{self.args.mines}_ep{self.agent.episodes}_winrate{win_rate:.1f}.pth")
                    # state_dict_optimizers = [worker.state_dict_optimizer() for worker in self.worker_agents]
                    self.agent.save(best_filename)
                    print(f"Best win rate of {win_rate:.1f} saved at episode {self.agent.episodes}.")
                    self.best_win_rate = win_rate

    def make_memory(self, env):
        # memorize one episode
        print(f"\nEpisode {self.agent.episodes}:")
        env.reset()
        state = env.get_normalized_state()
        total_rewards = []
        reveald_cnt = 0
        failed_cnt = 0
        valid_actions = [(r, c) for r in range(env.rows) for c in range(env.cols)]
        skipped_cells = set()
        while not env.check_win() and failed_cnt <= self.MAX_RETRY:
            if env.first_click:
                # force to make a random action if first click
                action, _ = self.agent.act(state, valid_actions, force_random=True)
            else:
                action, _ = self.agent.act(state, valid_actions, safe_action_first=True, env=env)
            # avoid wrong action made again.
            valid_actions.remove(action)
            row, col = action
            lose, _ = env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=self.allow_retry)
            reward, done, lose, reveald_cnt, failed_cnt = self._calc_reward(self.agent, env, row, col, lose, failed_cnt, reveald_cnt,
                                                                            skipped_cells)
            next_state = env.get_normalized_state()
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

    @staticmethod
    def worker_train_episode(id, model, gamma, state, action, reward, next_state, done):
        # model = queue_arg.get()
        worker_agent = DQNWorkerAgent(id, gamma, model)
        grads, loss = worker_agent.train(state, action, reward, next_state, done)
        # queue_ret.put((id, grads, loss))
        # signal.wait()
        return grads, loss

    # 传参数版
    # def _train_episode(self):
    #     w_locals = []
    #     loss_locals = []
    #     reward_locals = []
    #     self.agent.model.cpu()
    #     print(f"Workers training...")
    #     with mp.Pool(self.args.num_processes) as pool:
    #         args = [(
    #             self.worker_agents[i],
    #             self.agent.memory,
    #             self.agent.model,
    #             # self.agent.lr_min,
    #             # self.agent.lr_decay,
    #             # self.agent.lr_decay_step,
    #             self.local_train_episodes,
    #             self.batch_size
    #         ) for i in range(self.args.num_processes)]
    #         for i, (w, loss, reward, epsilon, lr, optimizer_state_dict) in enumerate(pool.starmap(self.worker_train_episode, args)):
    #             w_locals.append(w)
    #             loss_locals.append(loss)
    #             reward_locals.append(reward)
    #             self.worker_agents[i].update_optimizer_from(optimizer_state_dict)
    #     # TODO 每次只有FedAvg时GPU利用率才会有一个高峰，多进程的时候GPU利用率上不去啊。
    #     print("Averaging models...")
    #     self.agent.model.to(self.agent.device)
    #     w_glob = FedAvg(w_locals, self.agent.device)
    #     avg_loss = np.average(loss_locals)
    #     avg_reward = np.average(reward_locals)
    #     self.agent.update_train_episode(w_glob, avg_loss, avg_reward, epsilon, lr)  # 在里面write_train_result, episode += 1

    # 传梯度版
    def _train_episode(self):
        grads_locals = []
        loss_locals = []
        # TODO remove moving model to cpu
        # TODO 把每个worker的model共享GPU内存，提高GPU利用率
        print(f"Workers training...")
        # queue_arg = mp.Queue()
        # queue_ret = mp.Queue()
        batches = random.sample(self.agent.memory, self.args.num_processes)
        # signals_finished = [mp.Event() for _ in range(self.args.num_processes)]
        # procs = []
        args = []
        with mp.Pool(self.args.num_processes) as pool:
            for i in range(self.args.num_processes):
                self.models_local[i].load_state_dict(self.agent.model.state_dict())
                # queue_arg.put(self.models_local[i])
                args.append((
                    i,
                    self.models_local[i],
                    # queue_arg,
                    # queue_ret,
                    # signals_finished[i],
                    self.agent.gamma,
                    *batches[i]
                ))
            for i, (grads, loss) in enumerate(pool.starmap(self.worker_train_episode, args)):
                grads_locals.append({name: grad for name, grad in grads.items()})
                loss_locals.append(loss)
                # signals_finished[i].set()
        #     procs.append(mp.Process(target=self.worker_train_episode, args=args))
        #
        # [p.start() for p in procs]
        #
        # # Retrieve the results from the queue
        # for _ in range(self.args.num_processes):
        #     i, grads, loss = queue_ret.get()
        #     grads_locals.append({name: grad.clone() for name, grad in grads.items()})
        #     loss_locals.append(loss)
        #     del grads, loss
        #     signals_finished[i].set()
        #
        # [p.join() for p in procs]

        print("Training global model...")
        grads_glob = GradAvg(grads_locals)
        avg_loss = np.average(loss_locals)
        self.agent.global_train_episode(grads_glob, avg_loss)

    def _test_play(self, test_episodes):
        game_loses = np.empty(test_episodes, dtype=bool)
        revealed_cnts = np.empty(test_episodes, dtype=int)
        rewards = np.empty(test_episodes, dtype=np.float64)
        losses = np.empty(test_episodes, dtype=np.float64)
        with mp.Pool(self.args.num_processes) as pool:
            args = [(self.agent.model, self.args.rows, self.args.cols, self.args.mines) for _ in range(test_episodes)]
            i = 0
            for game_lose, revealed_cnt, reward, loss in pool.starmap(self._worker_test_episode, args):
                game_loses[i] = game_lose
                revealed_cnts[i] = revealed_cnt
                rewards[i] = reward
                losses[i] = loss
                i += 1
        win_rate = np.average(game_loses == False)
        avg_revealed_prcnt = np.average(revealed_cnts) / (self.args.rows * self.args.cols - self.args.mines) * 100
        avg_reward = np.average(rewards)
        avg_loss = np.average(losses)
        self.agent.write_test_result(win_rate, avg_revealed_prcnt, avg_reward, avg_loss)
        print(f"Test: Win rate: {win_rate:.2f}, Avg revealed%: {avg_revealed_prcnt:.2f}, Avg reward: {avg_reward:.1f}, Avg loss: {avg_loss:.5f}")
        return win_rate

    @staticmethod
    def _worker_test_episode(model, rows, cols, mines):
        env = MinesweeperEnv(rows=rows, cols=cols, mines=mines)
        agent = FedDQNAgent(comment='', model=model, eval=True)
        agent.model.eval()
        state = env.get_normalized_state()
        rewards = []
        reveald_cnt = 0
        valid_actions = [(r, c) for r in range(env.rows) for c in range(env.cols)]
        skipped_cells = set()
        lose = False
        losses = []
        while not env.check_win() and not lose:
            if env.first_click:
                # force to make a random action if first click
                action, _ = agent.act(state, valid_actions, force_random=True)
            else:
                # action, _ = agent.act(state, valid_actions, safe_action_first=True, env=env)
                action, _ = agent.act(state, valid_actions)
            # avoid wrong action made again.
            valid_actions.remove(action)
            row, col = action
            lose, _ = env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=False,
                                         allow_retry=False)
            reward, done, lose, reveald_cnt, _ = ExperimentFed._calc_reward(agent, env, row, col, lose, 0, reveald_cnt,
                                                                            skipped_cells)
            next_state = env.get_normalized_state()
            # agent.remember(state, action, reward, next_state, done)
            rewards.append(reward)
            with torch.no_grad():
                loss = agent.calc_loss(state, action, reward, next_state, done).item()
            losses.append(loss)
            if lose:
                # game over
                break
            state = next_state
        avg_loss = np.average(losses)
        avg_reward = np.average(rewards)
        return lose, reveald_cnt, avg_reward, avg_loss

    @staticmethod
    def _calc_reward(agent, env, row, col, lose, failed_cnt, reveald_cnt, skipped_cells: set):
        if lose:
            failed_cnt += 1
            done = True
            reward = -agent.MAX_REWARD
        else:
            reveald_cnt += 1
            done = env.check_win()
            total_reward_prcnt = 0.1
            reward = agent.MAX_REWARD * total_reward_prcnt
        return reward, done, lose, reveald_cnt, failed_cnt

    def print_status(self, *args):
        failed_cnt, reveald_cnt, reward = args
        sys.stdout.write \
            (f"\nEp: {self.agent.episodes} \t Revealed: {reveald_cnt} \t Reward: {reward:.1f}\t\tFailed: {failed_cnt} \tEpsilon: {self.agent.epsilon:.2f}")


if __name__ == '__main__':
    exp = ExperimentFed()
    exp.run()

