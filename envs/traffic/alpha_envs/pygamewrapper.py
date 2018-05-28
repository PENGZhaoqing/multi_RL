import pygame
import numpy as np
from pygame.constants import KEYDOWN, KEYUP, K_F15


class PyGameWrapper(object):

    def __init__(self, width, height, actions={}):

        # Required fields
        self.actions = actions  # holds actions
        self.actions = actions

        self.score = 0.0  # required.
        self.lives = 0  # required. Can be 0 or -1 if not required.
        self.screen = None  # must be set to None
        self.clock = None  # must be set to None
        self.height = height
        self.width = width
        self.screen_dim = (width, height)  # width and height
        self.allowed_fps = None  # fps that the game-v0.03-v0.01 is allowed to run at.
        self.NOOP = K_F15  # the noop key
        self.rng = None

        self.rewards = {
            "positive": 1.0,
            "negative": -0.5,
            "tick": 0.0,
            "loss": -5.0,
            "win": 5.0
        }

    def setup(self):
        """
        Setups up the pygame env, the display and v3-v0.01 clock.
        """
        pygame.init()
        self.screen = pygame.display.set_mode(self.get_screen_dims(), 0, 32)
        self.clock = pygame.time.Clock()

    def set_actions(self, actions):

        for idx, action in enumerate(actions):
            kd = pygame.event.Event(KEYDOWN, {"key": action})
            pygame.event.post(kd)

    def draw_frame(self, draw_screen):
        """
        Decides if the screen will be drawn too
        """

        if draw_screen:
            pygame.display.update()

    def get_screen_rgb(self):
        """
        Returns the current v3-v0.01 screen in RGB format.

        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).

        """

        return pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)

    def tick(self, fps):
        """
        This sleeps the v3-v0.01 to ensure it runs at the desired fps.
        """
        return self.clock.tick_busy_loop(fps)

    def adjustRewards(self, rewards):
        """

        Adjusts the rewards the v3-v0.01 gives the agent

        Parameters
        ----------
        rewards : dict
            A dictonary of reward events to float rewards. Only updates if key matches those specificed in the init function.

        """
        for key in rewards.keys():
            if key in self.rewards:
                self.rewards[key] = rewards[key]

    def setRNG(self, rng):
        """
        Sets the rng for games.
        """

        if self.rng is None:
            self.rng = rng

    def get_game_state(self):
        """
        Gets a non-visual state representation of the v3-v0.01.

        Returns
        -------
        dict or None
            dict if the v3-v0.01 supports it and None otherwise.

        """
        return None

    def get_screen_dims(self):
        """
        Gets the screen dimensions of the v3-v0.01 in tuple form.

        Returns
        -------
        tuple of int
            Returns tuple as follows (width, height).

        """
        return self.screen_dim

    def get_actions(self):
        """
        Gets the actions used within the v3-v0.01.

        Returns
        -------
        list of `pygame.constants`

        """
        return self.actions.values()

    def init(self):
        """
        This is used to initialize the v3-v0.01, such reseting the score, lives, and player position.

        This is v3-v0.01 dependent.

        """
        raise NotImplementedError("Please override this method")

    def reset(self):
        """
        Wraps the init() function, can be setup to reset certain poritions of the v3-v0.01 only if needed.
        """
        self.init()

    def get_score(self):
        """
        Return the current score of the v3-v0.01.


        Returns
        -------
        int
            The current reward the agent has received since the last init() or reset() call.
        """
        raise NotImplementedError("Please override this method")

    def game_over(self):
        """
        Gets the status of the v3-v0.01, returns True if v3-v0.01 has hit a terminal state. False otherwise.

        This is v3-v0.01 dependent.

        Returns
        -------
        bool

        """
        raise NotImplementedError("Please override this method")

    def step(self, dt):
        """
        This method steps the v3-v0.01 forward one step in time equal to the dt parameter. The v3-v0.01 does not run unless this method is called.

        Parameters
        ----------
        dt : integer
            This is the amount of time elapsed since the last frame in milliseconds.

        """
        raise NotImplementedError("Please override this method")
