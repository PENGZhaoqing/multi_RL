import sys
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../')
from envs.pygamewrapper import PyGameWrapper
from agent import Agent
from utils import *
from pygame.constants import *
import pygame
import sys
import Queue

white = (255, 255, 255)
red = (255, 0, 0)
dark = (0, 0, 0)
yellow = (255, 255, 0)
blue = (0, 0, 255)

Key_mapping = {
    0: {"accelerate": K_q, "brake": K_a},
    1: {"accelerate": K_w, "brake": K_s},
    2: {"accelerate": K_e, "brake": K_d},
    3: {"accelerate": K_r, "brake": K_f},
    4: {"accelerate": K_t, "brake": K_g},
    5: {"accelerate": K_y, "brake": K_h},
    6: {"accelerate": K_u, "brake": K_j},
    7: {"accelerate": K_i, "brake": K_k},
    8: {"accelerate": K_o, "brake": K_l},
    9: {"accelerate": K_p, "brake": K_m},
    10: {"accelerate": K_z, "brake": K_x},
    11: {"accelerate": K_c, "brake": K_v},
    12: {"accelerate": K_b, "brake": K_n},
    13: {"accelerate": K_1, "brake": K_2},
    14: {"accelerate": K_3, "brake": K_4},
    15: {"accelerate": K_5, "brake": K_6},
    16: {"accelerate": K_7, "brake": K_8},
    17: {"accelerate": K_9, "brake": K_0},
    18: {"accelerate": K_UP, "brake": K_DOWN},
    19: {"accelerate": K_LEFT, "brake": K_RIGHT},
    20: {"accelerate": K_F1, "brake": K_F2},
}


class Entrance():
    def __init__(self, loc, type, avbl=True):
        self.loc = loc[::-1]
        self.type = type
        self.avbl = avbl


class TrafficSim(PyGameWrapper):
    def __init__(self, draw=False, width=48, height=48, agent_num=10, prob=0.05, mode='esay'):
        self.actions = {k: Key_mapping[k] for k in sorted(Key_mapping.keys())[:agent_num]}
        PyGameWrapper.__init__(self, width, height, actions=self.actions)
        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        self.draw = draw
        self.BG_COLOR = dark
        self.map = None
        self.mode = mode
        self.max_agent_num = agent_num
        self.agent_num = agent_num
        self.agents = pygame.sprite.Group()
        self.agents_dict = {}
        self.reward = np.zeros(self.max_agent_num)
        self.init_flag = False
        self.ids = None
        self.vis = False
        self.crosses = None
        self.info = np.zeros((self.max_agent_num, self.max_agent_num), dtype=int)
        self.info2 = np.zeros((self.max_agent_num, self.max_agent_num), dtype=int)
        self.ents = None
        self.prob = prob
        self.vis_range = (1, 1)
        self.range = (self.vis_range[0] * 2 + 1, self.vis_range[0] * 2 + 1)
        self.observation = np.zeros((self.max_agent_num, self.range[0] * self.range[1] * self.max_agent_num * 4))
        self.reset_agents = Queue.Queue()

    def get_score(self):
        return self.score

    def game_over(self):
        return False

    def init(self):

        self.font = pygame.font.SysFont("monospace", 13)

        if self.mode == 'hard':
            self.map = np.array([[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
        else:
            self.map = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], ])
        self.maze = np.zeros(self.map.shape + (self.max_agent_num, 4), dtype=int)
        height, width = self.map.shape
        self.bin = self.height / float(self.map.shape[0])

        self.ents = []
        for j in range(height - 1):
            if self.map[j - 1][0] == 0 and self.map[j][0] == 0:
                self.ents.append(Entrance(loc=[j, 0], type=3))
            elif self.map[j][width - 1] == 0 and self.map[j + 1][width - 1] == 0:
                self.ents.append(Entrance(loc=[j, width - 1], type=2))

        for i in range(width - 1):
            if self.map[0][i] == 0 and self.map[0][i + 1] == 0:
                self.ents.append(Entrance(loc=[0, i], type=0))
            elif self.map[height - 1][i - 1] == 0 and self.map[height - 1][i] == 0:
                self.ents.append(Entrance(loc=[height - 1, i], type=1))

        self.crosses = []
        if self.mode == 'hard':
            self.crosses.append([4, 4])
            self.crosses.append([12, 4])
            self.crosses.append([12, 12])
            self.crosses.append([4, 12])
        else:
            self.crosses.append([6, 6])

    def reset(self):
        self.maze[:] = 0
        self.agents.empty()
        self.agents_dict = {}
        self.ids = range(self.agent_num)
        self.rng.shuffle(self.ids)
        self.reset_agents.queue.clear()

        if self.vis:
            if 0 in self.ids:
                self.ids.remove(0)
                car = self._new_car(0, [5, 7], 3)
            else:
                car = self.agents_dict[0]
                car.reset([5, 7], 3)
            car.cross_type = 1
            self.agents.add(car)
            self.agents_dict[car.id] = car
            self.maze[car.loc[1]][car.loc[0]][car.id][car.route()] = 1

            if 1 in self.ids:
                self.ids.remove(1)
                car = self._new_car(1, [6, 6], 0)
            else:
                car = self.agents_dict[1]
                car.reset([6, 6], 0)
            car.cross_type = 1
            self.agents.add(car)
            self.agents_dict[car.id] = car
            self.maze[car.loc[1]][car.loc[0]][car.id][car.route()] = 1

            locs = [[6, 2], [6, 10], [10, 6], [3, 6], [8, 7], [7, 1], [7, 13], [12, 7]]
            types = [0, 0, 2, 2, 3, 1, 1, 3]
            cnt = 0
            tmp = list(self.ids)
            for idx in tmp:
                self.ids.remove(idx)
                car = self._new_car(idx, locs[cnt], types[cnt])
                self.agents.add(car)
                self.agents_dict[car.id] = car
                self.maze[car.loc[1]][car.loc[0]][car.id][car.route()] = 1
                cnt += 1


    # @profile
    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                for idx, actions in self.actions.iteritems():
                    if self.agents_dict.has_key(idx):
                        agent = self.agents_dict[idx]
                        if key == actions["accelerate"]:
                            agent.dx = 1
                        if key == actions["brake"]:
                            agent.dx = 0

    def _new_car(self, id, loc, type):
        car = Agent(id, yellow, self.bin, 1, self.width, self.height, self.crosses)
        car.reset(loc, type)
        return car

    def _car_is_in_ent(self, car, ent):
        return car.loc == ent.loc

    def _is_collision(self, agent1, agent2):
        return agent1.loc == agent2.loc

    # @profile
    def step(self, dt):

        self.reward[:] = 0.0
        self.maze[:] = 0
        self._handle_player_events()
        self.info2[:] = 0

        if self.game_over():
            return self.reward

        # dt /= 1000.0 if not dt == 1000 else 1000
        dt /= 1000

        # reset entrances
        for ent in self.ents:
            ent.avbl = True
            for car in self.agents:
                if self._car_is_in_ent(car, ent):
                    ent.avbl = False

        # reset the car which is out of map
        if not self.reset_agents.empty():
            self.rng.shuffle(self.ents)
            for ent in self.ents:
                if ent.avbl is True and not self.reset_agents.empty():
                    car = self.reset_agents.get()
                    car.reset(ent.loc, ent.type)
                    ent.avbl = False

        # add new car if not reaching capacity
        if len(self.ids) > 0:
            for ent in self.ents:
                if ent.avbl is True and len(self.ids) > 0 and self.rng.rand() <= self.prob:
                    car = self._new_car(self.ids[0], ent.loc, ent.type)
                    self.agents.add(car)
                    self.agents_dict[car.id] = car
                    ent.avbl = False
                    del self.ids[0]

        self.agents.update(dt)
        self.screen.fill(self.BG_COLOR)

        # reset cars if they are out of maze
        for car in self.agents.sprites():
            if car.out_of_maze:
                continue
            if car.loc[0] < 0 or car.loc[0] > self.map.shape[1] - 1 \
                    or car.loc[1] < 0 or car.loc[1] > self.map.shape[0] - 1:
                car.out_of_maze = True
                self.reset_agents.put(car)
            else:
                # set agents loc in maze
                assert type(car.loc[1]) == int and type(car.loc[0]) == int
                self.maze[car.loc[1]][car.loc[0]][car.id][car.route()] = 1

                if car.dx == 0:
                    # draw blue rect when car brakes
                    if self.draw:
                        pygame.draw.circle(self.screen, blue, car.rect.center, int(car.radius), 0)
                else:
                    # get reward if rush
                    self.reward[car.id] += self.rewards['tick'] * car.steps

                if self.draw:
                    # draw the agent id
                    # label = self.font.render(str(car.id) + str(car.loc), 1, white)
                    label = self.font.render(str(car.id), 1, white)
                    # self.screen.blit(label, (car.rect.center[0] - 5, car.rect.center[1] - 5))

                # check collisions
                for other in self.agents.sprites():
                    if car != other and self._is_collision(car, other):
                        self.reward[car.id] += self.rewards['negative']
                        if self.draw:
                            x = (car.rect.center[0] + other.rect.center[0]) / 2.0
                            y = (car.rect.center[1] + other.rect.center[1]) / 2.0
                            pygame.draw.circle(self.screen, red, (int(x), int(y)), int(car.radius), 0)

        if self.draw:
            # render agent
            self.agents.draw(self.screen)
            # render roads
            shape = np.asarray(self.map).shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if self.map[i][j] == 0:
                        pygame.draw.rect(self.screen, white,
                                         (self.bin * j, self.bin * i, self.bin, self.bin), 1)

        return self.reward, self.info

    def in_observation(self, agent, other):
        return np.abs(agent.loc[0] - other.loc[0]) <= self.vis_range[0] and np.abs(agent.loc[1] - other.loc[1]) <= \
               self.vis_range[1]


    def get_game_state(self):
        self.observation[:] = 0.0
        self.info2[:] = 0
        tmp = np.pad(self.maze, [(1, 1), (1, 1), (0, 0), (0, 0)], mode='constant')
        # print tmp.shape
        for agent in self.agents.sprites():
            if not agent.out_of_maze:
                x, y = agent.loc
                dx, dy = self.vis_range
                self.observation[agent.id] = tmp[y:y + 2 * dy + 1, x:x + 2 * dx + 1].flatten()
        assert self.observation.shape == (
            self.max_agent_num, self.range[0] * self.range[1] * self.max_agent_num * 4)

        for agent in self.agents.sprites():
            if not agent.out_of_maze:
                for other_agent in self.agents.sprites():
                    if not other_agent.out_of_maze:
                        if agent == other_agent: continue
                        if self.in_observation(agent, other_agent):
                            self.info2[agent.id][other_agent.id] = 1

        return self.observation

        # def get_game_state(self):
        #     self.observation[:] = 0.0
        #     for agent in self.agents.sprites():
        #         if not agent.out_of_maze:
        #             x, y = agent.loc
        #             # feature = [(agent.steps + 1) / float(self.map.shape[0])]
        #             # feature = [1]
        #             dx, dy = self.vis_range
        #             tmp = np.zeros((self.range[0], self.range[1], self.max_agent_num, 4), dtype=int)
        #             for i in range(x - dx, x + dx + 1):
        #                 for j in range(y - dy, y + dy + 1):
        #                     if 0 < i < self.maze.shape[1] and 0 < j < self.maze.shape[0]:
        #                         agent_idxs = np.argwhere(self.maze[j][i] == 1).flatten()
        #                         if len(agent_idxs) > 0:
        #                             for id in agent_idxs:
        #                                 other = self.agents_dict[id]
        #                                 # if other == agent: continue
        #                                 other_move = other.route()
        #                                 tmp[i - x + dx][j - y + dy][id][other_move] = 1
        #             # feature.extend(tmp.flatten())
        #             self.observation[agent.id] = tmp.flatten()
        #     assert self.observation.shape == (
        #         self.max_agent_num, self.range[0] * self.range[1] * self.max_agent_num * 4)
        #     return self.observation


if __name__ == "__main__":
    import numpy as np
    import time

    rewards = {
        "positive": 1.0,
        "negative": -10,
        "tick": 0.0001,
        "loss": -5.0,
        "win": 5.0
    }

    pygame.init()
    game = TrafficSim(width=512, height=512, agent_num=15, draw=True, prob=0.25, mode='hard')
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rewards = rewards
    game.rng = np.random.RandomState(24)
    game.init()
    game.reset()

    while True:
        start = time.time()
        # dt = game.clock.tick_busy_loop(1)
        if game.game_over():
            game.init()
        reward = game.step(1000)
        game.get_game_state()
        # print reward
        pygame.display.update()
        end = time.time()
        print 1 / (end - start)
