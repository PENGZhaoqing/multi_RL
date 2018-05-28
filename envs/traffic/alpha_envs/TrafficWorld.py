from pygamewrapper import PyGameWrapper
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
    def __init__(self, draw=False, width=48, height=48, agent_num=10):
        self.actions = {k: Key_mapping[k] for k in sorted(Key_mapping.keys())[:agent_num]}
        PyGameWrapper.__init__(self, width, height, actions=self.actions)
        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        self.draw = draw
        self.BG_COLOR = dark
        self.bin = height / 14.0
        self.agent_num = agent_num
        self.agents = pygame.sprite.Group()
        self.agents_dict = {}
        self.reward = np.zeros(self.agent_num)
        self.init_flag = False
        self.observation = np.zeros((self.agent_num, (9 * (1 + 4) + 5)))
        self.id = 0
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
        self.ents = []
        self.prob = 0.25
        self.vis_range = (1, 1)
        self.reset_agents = Queue.Queue()
        self.move_oh = self._create_onehot(4)
        self.id_oh = self._create_onehot(self.agent_num)
        self.maze = np.zeros(self.map.shape, dtype=int)

    def get_score(self):
        return self.score

    def game_over(self):
        return False

    def init(self):
        self.font = pygame.font.SysFont("monospace", 15)
        assert self.init_flag == False, "Init Game Twice!!!"
        height, width = self.map.shape

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

        self.init_flag = True

    def reset(self):
        assert self.init_flag, "Init Game First"
        self.maze[:] = -1
        self.agents.empty()
        self.agents_dict = {}
        self.id = 0
        self.reset_agents.queue.clear()

        # add new car if not reaching capacity
        if self.id < self.agent_num:
            for ent in self.ents:
                if ent.avbl is True and self.id < self.agent_num and self.rng.rand() <= self.prob:
                    car = self._new_car(self.id, ent.loc, ent.type)
                    self.agents.add(car)
                    self.agents_dict[car.id] = car
                    ent.avbl = False
                    self.id += 1

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

    def _create_onehot(self, size):
        onehot = np.zeros((size, size), dtype=int)
        for i in range(size):
            onehot[i, i] = 1
        return onehot

    def _new_car(self, id, loc, type):
        car = Agent(id, yellow, self.bin, 1, self.width, self.height)
        car.reset(loc, type)
        return car

    def _car_is_in_ent(self, car, ent):
        x1, y1 = car.loc
        x2, y2 = ent.loc
        if x1 == x2 and y1 == y2:
            return True
        else:
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 1 - 0.001

    def _is_collision(self, agent1, agent2):
        x1, y1 = agent1.loc
        x2, y2 = agent2.loc
        if x1 == x2 and y1 == y2:
            return True
        else:
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 1 - 0.001

    # @profile
    def step(self, dt):

        self.reward[:] = 0.0
        self.maze[:] = -1

        if self.game_over():
            return self.reward

        # dt /= 1000.0 if not dt == 1000 else 1000
        dt /= 1000

        self._handle_player_events()
        self.agents.update(dt)
        self.screen.fill(self.BG_COLOR)

        # reset entrances
        for ent in self.ents:
            ent.avbl = True
            for car in self.agents:
                if self._car_is_in_ent(car, ent):
                    ent.avbl = False

        if not self.reset_agents.empty():
            self.rng.shuffle(self.ents)
            for ent in self.ents:
                if ent.avbl is True and not self.reset_agents.empty():
                    car = self.reset_agents.get()
                    car.reset(ent.loc, ent.type)
                    ent.avbl = False

        # add new car if not reaching capacity
        if self.id < self.agent_num:
            for ent in self.ents:
                if ent.avbl is True and self.id < self.agent_num and self.rng.rand() <= self.prob:
                    car = self._new_car(self.id, ent.loc, ent.type)
                    self.agents.add(car)
                    self.agents_dict[car.id] = car
                    ent.avbl = False
                    self.id += 1

        # reset cars if they are out of maze
        for car in self.agents.sprites():
            if car.out_of_maze:
                continue
            if car.pos[0] < 0 or car.pos[0] > self.width or car.pos[1] < 0 or car.pos[1] > self.height:
                car.out_of_maze = True
                self.reset_agents.put(car)
            else:
                # set agents loc in maze
                assert type(car.loc[1]) == int and type(car.loc[0]) == int
                self.maze[car.loc[1]][car.loc[0]] = car.id

        for agent in self.agents.sprites():
            if not agent.out_of_maze:
                if agent.dx == 0:
                    # draw the agent id
                    pygame.draw.circle(self.screen, blue, agent.rect.center, int(agent.radius), 0)
                else:
                    self.reward[agent.id] += self.rewards['tick'] * agent.steps
                # draw blue rect when car brakes
                label = self.font.render(str(agent.id) + str(agent.loc), 1, white)
                self.screen.blit(label, agent.rect.center)

        # check collision
        for car in self.agents.sprites():
            if not car.out_of_maze:
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

        return self.reward

    def get_game_state(self):
        self.observation[:] = 0
        for agent in self.agents.sprites():
            if not agent.out_of_maze:
                x, y = agent.loc
                tmp = list(self.move_oh[agent.move])
                tmp.extend([agent.steps/14.0])
                dx, dy = self.vis_range
                for i in range(x - dx, x + dx + 1):
                    for j in range(y - dy, y + dy + 1):
                        if i > 0 and i < self.maze.shape[1] and j > 0 and j < self.maze.shape[0] and self.maze[j][
                            i] > 0:
                            id = self.maze[j][i]
                            # tmp.extend(list(self.id_oh[id]))
                            tmp.extend([1])
                            tmp.extend(list(self.move_oh[self.agents_dict[id].move]))
                        else:
                            tmp.extend([0] * (1 + 4))
                self.observation[agent.id] = tmp
            assert self.observation.shape == (self.agent_num, (9 * (1 + 4) + 5))
        return self.observation


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
    game = TrafficSim(width=512, height=512, agent_num=5, draw=True)
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rewards = rewards
    game.rng = np.random.RandomState(24)
    game.init()
    game.reset()

    while True:
        start = time.time()
        dt = game.clock.tick_busy_loop(1)
        if game.game_over():
            game.init()
        reward = game.step(dt)
        # print reward
        pygame.display.update()
        end = time.time()
        print game.get_game_state().shape
        # print 1 / (end - start)
        # if v3-v0.01.getScore() > 0:
        # print "Score: {:0.3f} ".format(v3-v0.01.getScore())
