from envs.pygamewrapper import PyGameWrapper
from agent import *
from utils import *
from pygame.constants import *
import pygame
from numpy import random
from math import sqrt, sin, cos
import sys

COLOR_MAP = {"white": (255, 255, 255),
             "hunter": (0, 0, 255),
             "prey": (20, 255, 20),
             "toxin": (255, 20, 20),
             'grey': (144, 144, 144)}

Key_mapping = {
    0: {"up": K_UP, "left": K_LEFT, "right": K_RIGHT, "down": K_DOWN},
    1: {"up": K_1, "left": K_2, "right": K_q, "down": K_w},
    2: {"up": K_3, "left": K_4, "right": K_e, "down": K_r},
    3: {"up": K_5, "left": K_6, "right": K_t, "down": K_y},
    4: {"up": K_7, "left": K_8, "right": K_u, "down": K_i},
    5: {"up": K_9, "left": K_0, "right": K_o, "down": K_p},
    6: {"up": K_a, "left": K_s, "right": K_z, "down": K_x},
    7: {"up": K_d, "left": K_f, "right": K_c, "down": K_v},
    8: {"up": K_g, "left": K_h, "right": K_b, "down": K_n},
    9: {"up": K_j, "left": K_k, "right": K_m, "down": K_COMMA}
}


class HunterWorld(PyGameWrapper):
    def __init__(self, draw=False, width=48, height=48, num_preys=10, num_hunters=3, num_toxins=10):

        self.actions = {k: Key_mapping[k] for k in sorted(Key_mapping.keys())[:num_hunters]}

        PyGameWrapper.__init__(self, width, height, actions=self.actions)
        self.draw = draw
        self.BG_COLOR = COLOR_MAP['white']
        self.EYES = 24

        self.MAX_HUNTER_NUM = num_hunters
        self.HUNTER_NUM = num_hunters
        self.HUNTER_COLOR = COLOR_MAP['hunter']
        self.HUNTER_SPEED = width * 0.3
        self.HUNTER_RADIUS = percent_round_int(width, 0.015)
        self.hunters = pygame.sprite.Group()
        self.hunters_dic = {}

        self.PREY_COLOR = COLOR_MAP['prey']
        self.PREY_SPEED = width * 0.25
        self.PREY_NUM = num_preys
        self.PREY_RADIUS = percent_round_int(width, 0.015)
        self.preys = pygame.sprite.Group()

        self.TOXIN_NUM = num_toxins
        self.TOXIN_COLOR = COLOR_MAP['toxin']
        self.TOXIN_SPEED = 0.25 * width
        self.TOXIN_RADIUS = percent_round_int(width, 0.015)
        self.toxins = pygame.sprite.Group()
        self.walls = None
        self.preys_num = num_preys
        self.toxins_num = num_toxins
        self.all_entities = None

        self.observation = np.zeros((self.MAX_HUNTER_NUM, self.EYES * (num_hunters + 3) + 2))
        self.reward = np.zeros(self.HUNTER_NUM)
        self.info = np.zeros((self.HUNTER_NUM, self.HUNTER_NUM), dtype=int)
        self.info2 = np.zeros((self.HUNTER_NUM, self.HUNTER_NUM), dtype=int)
        self.onehot = self.agent_one_hot(self.HUNTER_NUM)

    def agent_one_hot(self, agent_num):
        tmp = np.zeros((agent_num, agent_num), dtype=int)
        for i in range(agent_num):
            tmp[i][i] = 1
        return tmp

    def _rand_postion(self, agents):
        pos = []
        for agent in agents:
            pos_x = random.uniform(agent.radius, self.width - agent.radius)
            pos_y = random.uniform(agent.radius, self.height - agent.radius)
            pos.append([pos_x, pos_y])

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = math.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                while dist <= (agents[i].radius + agents[j].radius):
                    pos[i][0] = random.uniform(agents[i].radius, self.width - agents[i].radius)
                    pos[i][1] = random.uniform(agents[i].radius, self.height - agents[i].radius)
                    dist = math.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
        return pos

    def _init_postions(self):
        pos_list = [list(self.rng.rand(2, )) for _ in range(50)]
        return [[pos_list[i][0] * self.width, pos_list[i][1] * self.height] for i in range(len(pos_list))]

    def _init_directions(self):
        dir_list = [list(self.rng.rand(2, ) - 0.5) for _ in range(50)]
        return [normalization(dir_list[i]) for i in range(len(dir_list))]

    def get_score(self):
        return self.score

    def game_over(self):
        return False

    def init(self):

        self.all_entities = []
        self.walls = []
        self.toxins.empty()
        self.preys.empty()
        self.hunters.empty()
        self.hunters_dic = {}
        self.font = pygame.font.SysFont("monospace", 18)

        # self.walls.append(
        #     Wall([self.width * 1 / 10, self.height / 2 - self.HUNTER_RADIUS * 8 / 2], self.width * 8 / 10,
        #          self.HUNTER_RADIUS * 8))

        self.walls = []
        self.FIX_DIR = self._init_directions()
        self.FIX_POS = self._init_postions()

        id = 0
        for _ in range(self.HUNTER_NUM):
            hunter = Hunter(id, self.HUNTER_RADIUS, self.HUNTER_COLOR,
                            self.HUNTER_SPEED, self.width, self.height, self.walls, type='hunter')
            hunter.init_direction((0, 0))
            hunter.init_positon(self.FIX_POS[id])
            self.hunters.add(hunter)
            self.hunters_dic[id] = hunter
            self.all_entities.append(hunter)
            id += 1
        for _ in range(self.preys_num):
            prey = Other(id, self.PREY_RADIUS, self.PREY_COLOR, self.PREY_SPEED,
                         self.width, self.height, self.walls, type='prey')
            prey.init_direction(self.FIX_DIR[id])
            prey.init_positon(self.FIX_POS[id])
            self.preys.add(prey)
            self.all_entities.append(prey)
            id += 1
        for _ in range(self.toxins_num):
            toxin = Other(id, self.TOXIN_RADIUS, self.TOXIN_COLOR, self.TOXIN_SPEED,
                          self.width, self.height, self.walls, type='toxin')
            toxin.init_direction(self.FIX_DIR[id])
            toxin.init_positon(self.FIX_POS[id])
            self.toxins.add(toxin)
            self.all_entities.append(toxin)
            id += 1

    def reset(self):
        for agent in self.all_entities:
            agent.reset_pos()
            agent.reset_orientation()

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                for idx, actions in self.actions.iteritems():
                    if self.hunters_dic.has_key(idx):
                        agent = self.hunters_dic[idx]

                        if key == actions["left"]:
                            agent.dx = -agent.speed
                            agent.dy = 0
                            agent.accelerate = True

                        if key == actions["right"]:
                            agent.dx = agent.speed
                            agent.dy = 0
                            agent.accelerate = True

                        if key == actions["up"]:
                            agent.dy = -agent.speed
                            agent.dx = 0
                            agent.accelerate = True

                        if key == actions["down"]:
                            agent.dy = agent.speed
                            agent.dx = 0
                            agent.accelerate = True

    # @profile
    def step(self, dt):

        self.reward[:] = 0.0
        self.info[:] = 0
        self.info2[:] = 0
        if self.game_over():
            return self.reward

        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)
        self._handle_player_events()

        for prey in self.preys:
            hunter_pair = []
            for hunter in self.hunters:
                if count_distant(prey, hunter) <= (hunter.range - prey.radius):
                    hunter_pair.append(hunter.id)
            if len(hunter_pair) >= 2:
                for hunter_id in hunter_pair:
                    if self.hunters_dic[hunter_id].dx == 0 and self.hunters_dic[hunter_id].dy == 0: continue
                    self.reward[hunter_id] += self.rewards["positive"]

                for id in hunter_pair:
                    for other_id in hunter_pair:
                        if id == other_id: continue
                        self.info[id][other_id] += 1

                # self.hungrey += self.rewards["positive"]
                prey.rand_orientation()
                prey.rand_pos()

        for hunter in self.hunters:
            for toxin in self.toxins:
                if count_distant(toxin, hunter) < (hunter.radius + toxin.radius):
                    # self.lives -= 1
                    toxin.rand_orientation()
                    toxin.rand_pos()
                    self.reward[hunter.id] += self.rewards["negative"]

        self.hunters.update(dt)
        self.preys.update(dt)
        self.toxins.update(dt)

        if self.draw:
            self.hunters.draw(self.screen)
            self.preys.draw(self.screen)
            self.toxins.draw(self.screen)

            for idx in range(self.HUNTER_NUM):
                label = self.font.render(str(idx), 1, (0, 0, 0))
                self.screen.blit(label,
                                 (self.hunters_dic[idx].rect.center[0] + 5, self.hunters_dic[idx].rect.center[1] + 5))

            for wall in self.walls:
                pygame.draw.rect(self.screen, COLOR_MAP['grey'], wall.rect)
            self.get_game_state()

        return self.reward, self.info

    def get_game_state(self):
        self.observation[:] = 0
        self.info2[:] = 0
        for i in range(self.HUNTER_NUM):
            hunter = self.hunters_dic[i]
            other_agents = []
            for j in range(len(self.all_entities)):
                agent = self.all_entities[j]
                if agent is hunter: continue
                if count_distant(agent, hunter) <= agent.radius + hunter.out_radius:
                    other_agents.append(agent)
            ob = self.observe1(hunter, other_agents)
            state = np.append(ob, [hunter.velocity[0] / self.width, hunter.velocity[1] / self.height])
            self.observation[i] = state[:]
        assert self.observation.shape == (self.MAX_HUNTER_NUM, self.EYES * (self.MAX_HUNTER_NUM + 3) + 2)
        return self.observation

    def observe1(self, hunter, others):
        center = list(hunter.rect.center)
        out_radius = hunter.out_radius - hunter.radius
        observation = np.zeros((self.EYES, 3 + self.MAX_HUNTER_NUM))
        angle = 2 * np.pi / self.EYES
        other_agents = others[:]
        for i in range(0, self.EYES):
            sin_angle = sin(angle * i)
            cos_angle = cos(angle * i)

            if i == 0:
                dis_to_wall = abs(hunter.pos[1] - hunter.radius)
                if dis_to_wall < out_radius:
                    observation[i][0] = 1 - dis_to_wall / out_radius
                    self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])
                else:
                    for wall in self.walls:
                        if wall.rect.left <= hunter.pos[0] <= wall.rect.right:
                            dis_to_wall = abs(hunter.pos[1] - hunter.radius - wall.rect.bottom)
                            if dis_to_wall < out_radius:
                                observation[i][0] = 1 - dis_to_wall / out_radius
                                self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])
            elif i == 6:
                dis_to_wall = abs(hunter.pos[0] + hunter.radius - self.width)
                if dis_to_wall < out_radius:
                    observation[i][0] = 1 - dis_to_wall / out_radius
                    self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])
                else:
                    for wall in self.walls:
                        if wall.rect.top <= hunter.pos[1] <= wall.rect.bottom:
                            dis_to_wall = abs(hunter.pos[0] + hunter.radius - wall.rect.left)
                            if dis_to_wall < out_radius:
                                observation[i][0] = 1 - dis_to_wall / out_radius
                                self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])

            elif i == 12:
                dis_to_wall = abs(hunter.pos[1] + hunter.radius - self.height)
                if dis_to_wall < out_radius:
                    observation[i][0] = 1 - dis_to_wall / out_radius
                    self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])
                else:
                    for wall in self.walls:
                        if wall.rect.left <= hunter.pos[0] <= wall.rect.right:
                            dis_to_wall = abs(hunter.pos[1] + hunter.radius - wall.rect.top)
                            if dis_to_wall < out_radius:
                                observation[i][0] = 1 - dis_to_wall / out_radius
                                self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])
            elif i == 18:
                dis_to_wall = abs(hunter.pos[0] - hunter.radius)
                if dis_to_wall < out_radius:
                    observation[i][0] = 1 - dis_to_wall / out_radius
                    self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])
                else:
                    for wall in self.walls:
                        if wall.rect.top <= hunter.pos[1] <= wall.rect.bottom:
                            dis_to_wall = abs(hunter.pos[0] - hunter.radius - wall.rect.right)
                            if dis_to_wall < out_radius:
                                observation[i][0] = 1 - dis_to_wall / out_radius
                                self.draw_line(center, sin_angle, cos_angle, hunter, dis_to_wall, COLOR_MAP["grey"])

            for agent in other_agents:
                dis = self.line_distance1(center, [sin_angle, -cos_angle], hunter.out_radius, list(agent.rect.center),
                                          agent.radius)
                if dis is not False:
                    dis = max(dis - hunter.radius, 0)
                    assert 0 <= dis <= out_radius, str(dis)
                    if agent.type == 'prey':
                        observation[i][1] = 1.0 - dis / out_radius
                        self.draw_line(center, sin_angle, cos_angle, hunter, dis, COLOR_MAP["prey"])
                    elif agent.type == 'toxin':
                        observation[i][2] = 1.0 - dis / out_radius
                        self.draw_line(center, sin_angle, cos_angle, hunter, dis, COLOR_MAP["toxin"])
                    elif agent.type == 'hunter':
                        self.info2[hunter.id][agent.id] += 1
                        observation[i][3 + agent.id] = 1.0 - dis / out_radius
                        self.draw_line(center, sin_angle, cos_angle, hunter, dis, COLOR_MAP["hunter"])
                    break

        return observation

    def draw_line(self, center, sin_angle, cos_angle, hunter, line, color):
        if self.draw:
            start_pos = [center[0] + sin_angle * hunter.radius, center[1] - cos_angle * hunter.radius]
            end_pos = [0, 0]
            end_pos[0] = start_pos[0] + int(sin_angle * line)
            end_pos[1] = start_pos[1] - int(cos_angle * line)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    # http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
    def line_distance1(self, seg_a, seg_v_unit, seg_v_len, circ_pos, circ_rad):
        pt_v = [circ_pos[0] - seg_a[0], circ_pos[1] - seg_a[1]]
        proj = pt_v[0] * seg_v_unit[0] + pt_v[1] * seg_v_unit[1]
        if proj <= 0 or proj >= seg_v_len:
            return False
        proj_v = [seg_v_unit[0] * proj, seg_v_unit[1] * proj]
        closest = [int(proj_v[0] + seg_a[0]), int(proj_v[1] + seg_a[1])]
        dist_v = [circ_pos[0] - closest[0], circ_pos[1] - closest[1]]
        offset = sqrt(dist_v[0] ** 2 + dist_v[1] ** 2)
        if offset >= circ_rad:
            return False
        le = sqrt(circ_rad ** 2 - int(offset) ** 2)
        re = [closest[0] - seg_a[0], closest[1] - seg_a[1]]
        # if sqrt(re[0] ** 2 + re[1] ** 2) - le < 0:
        #     a = 1
        #     print a

        return sqrt(re[0] ** 2 + re[1] ** 2) - le


if __name__ == "__main__":
    import numpy as np
    import time

    pygame.init()
    game = HunterWorld(width=512, height=480, num_preys=20, num_hunters=5, draw=True)
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    game.reset()

    while True:
        start = time.time()
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.init()
        reward = game.step(dt)
        pygame.display.update()
        end = time.time()
        game.get_game_state()
        # print game.cooperative
        # print 1 / (end - start)
        # if v3-v0.01.getScore() > 0:
        # print "Score: {:0.3f} ".format(v3-v0.01.getScore())
