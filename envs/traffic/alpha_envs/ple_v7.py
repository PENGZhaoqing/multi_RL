import cv2
import numpy as np
import pygame
from PIL import Image
from pygamewrapper import PyGameWrapper
import matplotlib.pyplot as plt


class PLE(object):
    def __init__(self,
                 game, fps=30, frame_skip=1, num_steps=4,
                 reward_values={}, force_fps=True, display_screen=False,
                 add_noop_action=True, rng=24,
                 resized_rows=84,
                 resized_cols=84):

        self.resized_rows = resized_rows
        self.resized_cols = resized_cols
        self.game = game
        self.fps = fps
        self.frame_skip = frame_skip
        self.NOOP = None
        self.num_steps = num_steps
        self.force_fps = force_fps
        self.display_screen = display_screen
        self.add_noop_action = add_noop_action
        self.screen_buffer = None
        self.frame_count = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.rng = 24
        if reward_values:
            self.game.adjustRewards(reward_values)

        if isinstance(self.game, PyGameWrapper):
            if isinstance(rng, np.random.RandomState):
                self.rng = rng
            else:
                self.rng = np.random.RandomState(rng)

            pygame.display.set_mode(self.game.get_screen_dims(), 0, 32)

        self.game.setRNG(self.rng)
        self.init()

        if game.allowed_fps is not None and self.fps != game.allowed_fps:
            raise ValueError("Game requires %dfps, was given %d." %
                             (game.allowed_fps, game.allowed_fps))

    def init(self):
        self.game.setup()
        self.game.init()

    def reset_game(self):
        self.episode_step = 0
        self.episode_reward = 0
        self.game.reset()

    def get_action_set(self):
        return self.game.actions

    def get_frame_number(self):
        return self.frame_count

    def game_over(self):
        return self.episode_step >= 100 or self.game.game_over()

    def score(self):
        return self.game.get_score()

    def lives(self):
        return self.game.lives

    def get_screen_rgb(self):
        return self.game.get_screen_rgb()

    def get_screen_gray_scale(self):

        frame = self.get_screen_rgb()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = 255 - gray_image
        return gray_image

    def get_resized_gray_scale(self):
        return cv2.resize(self.get_screen_gray_scale(), (self.resized_cols, self.resized_rows),
                          interpolation=cv2.INTER_LINEAR)

    def save_screen(self, filename):
        frame = Image.fromarray(self.get_screen_rgb())
        frame.save(filename)

    def get_screen_dims(self):
        return self.game.get_screen_dims()

    def _draw_frame(self):
        self.game.draw_frame(self.display_screen)

    def get_states(self):
        return self.game.get_game_state()

    def act(self, actions):

        self.game.set_actions(actions)

        reward = np.zeros(self.game.reward.size)
        for i in range(self.num_steps):
            time_elapsed = self._tick()
            reward += self.game.step(time_elapsed)
            self._draw_frame()

        ob = self.get_states()
        terminal_flag = self.game_over()

        self.frame_count += self.num_steps
        self.episode_step += 1
        self.episode_reward += reward

        return ob, reward, terminal_flag

    def _tick(self):
        if self.force_fps:
            return 1000 / self.fps
            # return 1000.0 / self.fps
        else:
            return self.game.tick(self.fps)
