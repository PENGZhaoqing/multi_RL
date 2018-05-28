from utils import normalization
import pygame
from numpy import random


class Agent(pygame.sprite.Sprite):
    def __init__(self, id, color, radius, speed, screen_width, screen_height):
        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.radius = radius
        self.image = None
        self.color = color
        self.rect = None
        self.pos = [0, 0]
        self.loc = [0, 0]
        self.dx = 0
        self.brake = True
        self.id = id
        self.out_of_maze = True
        self.move = None
        self.steps = 0
        self.speed = speed
        self.image = pygame.Surface([self.radius, self.radius])
        self.image.fill(color)
        # self.image.set_colorkey((0, 0, 0))
        self.image.set_alpha(int(255 * 0.75))
        # pygame.draw.rect(self.image, color, [0, 0, self.radius, self.radius])

        # self.image = image.convert()
        self.rect = self.image.get_rect()

    def reset(self, loc, type):
        self.move = type
        self.loc = loc[:]
        self.out_of_maze = False
        self._set_pos(self.loc)
        self.steps = 0

    def _set_pos(self, loc):
        self.pos[0] = loc[0] * self.radius + self.radius / 2
        self.pos[1] = loc[1] * self.radius + self.radius / 2

    def update(self, dt):

        if self.dx > 0:
            self.steps += 1

        # 0 for down
        if self.move == 0:
            self.loc[1] += self.speed * dt * self.dx
        # 1 for up
        elif self.move == 1:
            self.loc[1] -= self.speed * dt * self.dx
        # 2 for left
        elif self.move == 2:
            self.loc[0] -= self.speed * dt * self.dx
        # 3 for right
        elif self.move == 3:
            self.loc[0] += self.speed * dt * self.dx

        # self.dx = 0
        self._set_pos(self.loc)
        self.rect.center = (self.pos[0], self.pos[1])

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)
