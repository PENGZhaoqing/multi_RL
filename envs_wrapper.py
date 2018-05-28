from envs.traffic.trafficworld import TrafficSim
from envs.hunter_envs.hunterworld import HunterWorld
from envs.ple import PLE
import numpy as np

traffic_rewards = {
    "positive": 1.0,
    "negative": -1,
    "tick": 0.01,
    "loss": -5.0,
    "win": 5.0
}

hunter_rewards = {
    "positive": 1.0,
    "negative": -1.0,
    "tick": -0.002,
    "loss": -2.0,
    "win": 2.0
}


class Env:
    def __init__(self, id, config):
        self.id = id
        self.config = config
        if config.game_name == 'traffic':
            self.game = TrafficSim(width=256, height=256, agent_num=config.max_agent_num, draw=False,
                                   prob=config.prob_start,
                                   mode=config.mode)
            self.ple = PLE(self.game, fps=1, force_fps=True, display_screen=False, rng=id,
                           reward_values=traffic_rewards, max_episodes=100,
                           resized_rows=80, resized_cols=80, num_steps=1)

        elif config.game_name == 'hunterworld':
            self.game = HunterWorld(width=256, height=256, num_preys=config.prey_num, draw=False,
                                    num_hunters=config.max_agent_num, num_toxins=config.toxin_num)
            self.ple = PLE(self.game, fps=30, force_fps=True, display_screen=False, reward_values=hunter_rewards,
                           resized_rows=80, resized_cols=80, num_steps=3, max_episodes=500, rng=id)

        self.state_dim = self.ple.get_states().shape
        self.max_agent_num = config.max_agent_num
        assert self.max_agent_num == self.state_dim[0]
        self.clear()
        self.done = False

    def append(self, obs, actions, values, rewards, alphas=None):
        self.obs = np.append(self.obs, obs, axis=1)
        self.values = np.append(self.values, values, axis=1)
        self.actions = np.append(self.actions, actions, axis=1)
        self.rewards = np.append(self.rewards, rewards, axis=1)
        if alphas is not None:
            self.alphas = np.append(self.alphas, alphas, axis=1)

    def clear(self):
        self.obs = [[] for _ in range(self.max_agent_num)]
        self.values = [[] for _ in range(self.max_agent_num)]
        self.rewards = [[] for _ in range(self.max_agent_num)]
        self.actions = [[] for _ in range(self.max_agent_num)]
        self.alphas = [[] for _ in range(self.max_agent_num)]

    def reset_var(self):
        self.clear()
        self.done = False

    # curriculum learning
    def reset_cirruculumn(self, epoch):
        mask = np.zeros((self.config.num_envs, self.config.max_agent_num), dtype=int)

        if self.config.game_name == 'traffic':
            agent_num = min(int(epoch / 2) + 2,
                            self.config.max_agent_num) if self.config.dynamic else self.config.max_agent_num
            self.game.prob = self.config.prob_start + (self.config.prob_end - self.config.prob_start) \
                                                      / self.config.epoch_num * epoch
            self.game.agent_num = agent_num
            self.game.vis = self.config.vis
            self.game.init()
        elif self.config.game_name == 'hunterworld':
            agent_num = min(int(epoch / 4) + 2, self.config.max_agent_num) \
                if self.config.dynamic else self.config.max_agent_num
            self.game.HUNTER_NUM = agent_num
            self.game.init()
        else:
            raise Exception

        mask[:, :agent_num] = 1

        return agent_num, mask
