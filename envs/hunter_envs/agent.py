from utils import normalization
import pygame
from numpy import random
import numpy as np


class Agent(pygame.sprite.Sprite):
    def __init__(self, id, radius, speed, screen_width, screen_height, walls, type):
        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.speed = speed
        self.radius = radius
        self.velocity = [0, 0]
        self.image = None
        self.rect = None
        self.pos = [0, 0]
        self.id = id
        self.init_pos = None
        self.init_dir = None
        self.walls = walls
        self.type = type

    def init_positon(self, pos):
        self.init_pos = [pos[0], pos[1]]

    def init_direction(self, velocity):
        self.init_dir = [velocity[0], velocity[1]]

    def reset_pos(self):
        self.pos[0] = self.init_pos[0]
        self.pos[1] = self.init_pos[1]

    def reset_orientation(self):
        self.velocity[0] = self.init_dir[0]
        self.velocity[1] = self.init_dir[1]

    def rand_pos(self):
        self.pos[0] = random.uniform(self.radius, self.SCREEN_WIDTH - self.radius)
        self.pos[1] = random.uniform(self.radius, self.SCREEN_HEIGHT - self.radius)

    def rand_orientation(self):
        self.velocity[0] = random.random() - 0.5
        self.velocity[1] = random.random() - 0.5
        self.velocity = normalization(self.velocity)

    def update(self, dt):
        pass

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Other(Agent):
    def __init__(self, id, radius, color, speed, screen_width, screen_height, walls, type):
        Agent.__init__(self, id, radius, speed, screen_width, screen_height, walls, type)
        image = pygame.Surface([radius * 2, radius * 2])
        image.set_colorkey((0, 0, 0))
        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )
        self.image = image.convert()
        self.rect = self.image.get_rect()

    def update(self, dt):
        new_x = self.pos[0] + self.velocity[0] * self.speed * dt
        new_y = self.pos[1] + self.velocity[1] * self.speed * dt

        flag = True
        for wall in self.walls:
            if wall.rect.left < new_x < wall.rect.right:
                if wall.rect.top < new_y - self.radius < wall.rect.bottom:
                    self.pos[1] = wall.rect.bottom + self.radius
                    self.velocity[1] = -self.velocity[1]
                    flag = False
                elif wall.rect.top < new_y + self.radius < wall.rect.bottom:
                    self.pos[1] = wall.rect.top - self.radius
                    self.velocity[1] = -self.velocity[1]
                    flag = False
            if wall.rect.top < new_y < wall.rect.bottom:
                if wall.rect.left < new_x - self.radius < wall.rect.right:
                    self.pos[0] = wall.rect.right + self.radius
                    self.velocity[0] = -self.velocity[0]
                    flag = False
                elif wall.rect.left < new_x + self.radius < wall.rect.right:
                    self.pos[0] = wall.rect.left - self.radius
                    self.velocity[0] = -self.velocity[0]
                    flag = False

        if flag:
            if new_x >= self.SCREEN_WIDTH - self.radius:
                self.pos[0] = self.SCREEN_WIDTH - self.radius
                self.velocity[0] = -self.velocity[0]
            elif new_x < self.radius:
                self.pos[0] = self.radius
                self.velocity[0] = -self.velocity[0]
            else:
                self.pos[0] = new_x

            if new_y >= self.SCREEN_HEIGHT - self.radius:
                self.pos[1] = self.SCREEN_HEIGHT - self.radius
                self.velocity[1] = -self.velocity[1]
            elif new_y < self.radius:
                self.pos[1] = self.radius
                self.velocity[1] = -self.velocity[1]
            else:
                self.pos[1] = new_y

        self.rect.center = (self.pos[0], self.pos[1])


class Hunter(Agent):
    def __init__(self, id, radius, color, speed, screen_width, screen_height, walls, type):
        Agent.__init__(self, id, radius, speed, screen_width, screen_height, walls, type)

        self.out_radius = radius * 8
        self.range = radius * 5
        image = pygame.Surface([self.radius * 2, self.radius * 2])
        image.set_colorkey((0, 0, 0))
        image.set_alpha(int(255 * 0.75))

        pygame.draw.circle(
            image,
            color,
            (self.radius, self.radius),
            radius,
            0
        )
        self.dx = 0
        self.dy = 0
        self.accelerate = True
        self.image = image.convert()
        self.rect = self.image.get_rect()

    def update(self, dt):

        if self.accelerate:
            self.velocity[0] = np.clip(self.velocity[0] + self.dx, a_min=-self.SCREEN_WIDTH * 0.3,
                                       a_max=self.SCREEN_WIDTH * 0.3)
            self.velocity[1] = np.clip(self.velocity[1] + self.dy, a_min=-self.SCREEN_WIDTH * 0.3,
                                       a_max=self.SCREEN_WIDTH * 0.3)
            self.accelerate = False

        # print self.velocity[0]
        new_x = self.pos[0] + self.velocity[0] * dt
        new_y = self.pos[1] + self.velocity[1] * dt

        flag = True
        for wall in self.walls:
            if wall.rect.left < new_x < wall.rect.right:
                if wall.rect.top < new_y - self.radius < wall.rect.bottom:
                    self.pos[1] = wall.rect.bottom + self.radius
                    self.velocity[1] = 0.0
                    flag = False
                elif wall.rect.top < new_y + self.radius < wall.rect.bottom:
                    self.pos[1] = wall.rect.top - self.radius
                    self.velocity[1] = 0.0
                    flag = False
            if wall.rect.top < new_y < wall.rect.bottom:
                if wall.rect.left < new_x - self.radius < wall.rect.right:
                    self.pos[0] = wall.rect.right + self.radius
                    self.velocity[0] = 0.0
                    flag = False
                elif wall.rect.left < new_x + self.radius < wall.rect.right:
                    self.pos[0] = wall.rect.left - self.radius
                    self.velocity[0] = 0.0
                    flag = False

        if flag:
            if new_x >= self.SCREEN_WIDTH - self.radius:
                self.pos[0] = self.SCREEN_WIDTH - self.radius
                self.velocity[0] = 0.0
            elif new_x < self.radius:
                self.pos[0] = self.radius
                self.velocity[0] = 0.0
            else:
                self.pos[0] = new_x
                self.velocity[0] *= 0.975

            if new_y >= self.SCREEN_HEIGHT - self.radius:
                self.pos[1] = self.SCREEN_HEIGHT - self.radius
                self.velocity[1] = 0.0
            elif new_y < self.radius:
                self.pos[1] = self.radius
                self.velocity[1] = 0.0
            else:
                self.pos[1] = new_y
                self.velocity[1] *= 0.975

        self.rect.center = (self.pos[0], self.pos[1])


class Wall(object):
    def __init__(self, pos, width, height):
        self.rect = pygame.Rect(pos[0], pos[1], width, height)
