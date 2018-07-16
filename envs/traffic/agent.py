from utils import normalization
import pygame
import numpy as np

random = np.random.RandomState(24)


class Agent(pygame.sprite.Sprite):
    def __init__(self, id, color, radius, speed, screen_width, screen_height, crosses):
        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.radius = radius
        self.image = None
        self.color = color
        self.rect = None
        self.crosses = crosses
        self.pos = [0, 0]
        self.loc = [0, 0]
        self.dx = 0
        self.brake = True
        self.id = id
        self.out_of_maze = True
        self.move = None
        self.cross_type = random.randint(3)  # 0,1,2
        self.steps = 0
        self.speed = speed
        self.in_cross = False
        self.to_cross = -1
        self.image = pygame.Surface([self.radius, self.radius])
        # self.image.fill(color)
        self.image.set_colorkey((0, 0, 0))
        self.image.set_alpha(int(255 * 0.8))
        pygame.draw.rect(self.image, color, (0, 0, radius, radius), 0)
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

    def route(self):
        move = self.move
        if self.to_cross > -1:
            move = self.to_cross
        elif self.in_cross:
            pass
        else:
            for cross in self.crosses:
                cross_i, cross_j = cross
                if self.loc == [cross_j, cross_i]:
                    if self.cross_type == 2:
                        move = 2
                elif self.loc == [cross_j + 1, cross_i]:
                    if self.cross_type == 2:
                        move = 1
                elif self.loc == [cross_j, cross_i + 1]:
                    if self.cross_type == 2:
                        move = 0
                elif self.loc == [cross_j + 1, cross_i + 1]:
                    if self.cross_type == 2:
                        move = 3
        return move

    def update(self, dt):

        if self.dx > 0:
            self.steps += 1

            if self.to_cross > -1:
                # go straight forward or turn left(relatively)
                self.move = self.to_cross
                self.to_cross = -1
                self.cross_type = random.randint(3)
            elif self.in_cross:
                # turn left(relatively)
                self.in_cross = False
                self.cross_type = random.randint(3)
            else:
                for cross in self.crosses:
                    cross_i, cross_j = cross
                    if self.loc == [cross_j, cross_i]:
                        if self.cross_type == 0:
                            self.to_cross = 3  # to right
                            self.in_cross = True
                        elif self.cross_type == 1:
                            self.to_cross = 0  # down
                        elif self.cross_type == 2:
                            self.move = 2  # left
                            self.cross_type = random.randint(3)
                        else:
                            raise Exception
                    elif self.loc == [cross_j + 1, cross_i]:
                        if self.cross_type == 0:
                            self.to_cross = 0  # to down
                            self.in_cross = True
                        elif self.cross_type == 1:
                            self.to_cross = 2  # left
                        elif self.cross_type == 2:
                            self.move = 1
                            self.cross_type = random.randint(3)
                        else:
                            raise Exception
                    elif self.loc == [cross_j, cross_i + 1]:
                        if self.cross_type == 0:
                            self.to_cross = 1  # to up
                            self.in_cross = True
                        elif self.cross_type == 1:
                            self.to_cross = 3  # right
                        elif self.cross_type == 2:
                            self.move = 0
                            self.cross_type = random.randint(3)
                        else:
                            raise Exception
                    elif self.loc == [cross_j + 1, cross_i + 1]:
                        if self.cross_type == 0:
                            self.to_cross = 2  # to left
                            self.in_cross = True
                        elif self.cross_type == 1:
                            self.to_cross = 1  # up
                        elif self.cross_type == 2:
                            self.move = 3
                            self.cross_type = random.randint(3)
                        else:
                            raise Exception

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
