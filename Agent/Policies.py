import logging
import random

import numpy as np

from Game.Card import Card
from Game.Card import cards_to_bool_array


class Policy(object):

    def __init__(self, estimator, epsilon, decay=1):
        self.logger = logging.getLogger('wizard-rl.Policies.Policy')
        self.estimator = estimator
        self.epsilon = epsilon
        self.decay_rate = decay
        # In case we need to reset epsilon.
        self.original_epsilon = epsilon

    def get_action(self, x):
        raise NotImplementedError("This needs to be implemented by"
                                  "your Policy class.")

    def get_playable_q(self, x):
        """
        Returns the q values calculated by the estimator, but filtered by
        the available cards in hand.
        Args:
            x: Game state, the first 54 elements should describe the hand
            of the player.

        Returns:
            q_playable: (np.array) the q values calculated by the estimator,
            any q corresponding to a card NOT playable by the user has been
            assigned the min q value.
        """

        playable_bool = cards_to_bool_array(x)
        # Get the Q value estimations from the estimator.
        q = self.estimator.predict(x)[0]
        # Filter so that not playable cards have the min Q values.
        q_playable = q.copy()
        q_playable[~playable_bool] = np.min(q) - 1
        return q_playable


class EGreedyPolicy(Policy):

    def __init__(self, estimator, epsilon, decay=0.999):
        super().__init__(estimator, epsilon, decay)
        self.exploration_steps = 0

    def get_action(self, x):
        """
        Returns the probabilities for each action
        Args:
            x: np.array the first 54 elements should describe the
            cards in the hand of the player.

        Returns:
            probs: (np.array) 1D array describing the probability of choosing
            each action available.

        """
        self.epsilon *= self.decay_rate
        # self.logger.info("Exploration with epsilon: {}".format(self.epsilon))
        num_a = Card.DIFFERENT_CARDS

        # All probabilities start as 0
        e = random.uniform(0, 1)

        if e < self.epsilon:
            # random choice
            playable_bool = cards_to_bool_array(x)
            playable_cards = np.arange(0, num_a)[playable_bool]
            action = np.random.choice(playable_cards)
        else:
            q_playable = self.get_playable_q(x)
            action = np.argmax(q_playable)
        return action
