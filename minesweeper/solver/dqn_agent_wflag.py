import numpy as np
import torch

from solver.dqn_agent import DQNAgent


class DQNAgentWFlag(DQNAgent):
    def _action_to_index(self, action, cols):
        row, col, flag = action
        return (row * cols + col) * 2 + flag

    @torch.no_grad()
    def test_play(self, env, test_episodes=100, allow_recursive=False):
        result = []
        total_revealed_prcnts = []
        total_flagged_prcnts = []
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
                    env.make_move(row, col, flag=True, allow_click_revealed_num=False, allow_recursive=allow_recursive,
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
                    lose, last_revealed_cells = env.make_move(row, col, flag=False, allow_click_revealed_num=False, allow_recursive=allow_recursive,
                                         allow_retry=False)
                    for r, c in last_revealed_cells:
                        valid_actions_reveal.remove((r, c, 0))
                        try:
                            valid_actions_flag.remove((r, c, 1))
                        except ValueError:
                            # 上一把flag错了已经去掉了这个格子
                            pass

                    reveald_cnt += len(last_revealed_cells)

            result.append(lose)
            total_revealed_prcnts.append(reveald_cnt / (env.rows * env.cols - env.mines) * 100)
            total_flagged_prcnts.append(correct_flagged_cnt / env.mines * 100)

        win_rate = result.count(False) / test_episodes
        avg_revealed_prcnt = np.average(total_revealed_prcnts)
        avg_flag_prcnt = np.average(total_flagged_prcnts)
        if not self.eval:
            self.writer.add_scalar('TEST: Win rate', win_rate, self.episodes)
            self.writer.add_scalar('TEST: Avg revealed percent', avg_revealed_prcnt, self.episodes)
            self.writer.add_scalar('TEST: Avg correct flag percent', avg_flag_prcnt, self.episodes)
        return win_rate, avg_revealed_prcnt, avg_flag_prcnt
