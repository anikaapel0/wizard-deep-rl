import numpy as np
from GameUtilities.Card import Card
from Estimators.PolicyEstimators import PolicyGradient
import logging


class Policy(object):

    def __init__(self, estimator):
        self.logger = logging.getLogger('wizard-rl.Policy')

        self.estimator = estimator

    def get_action(self, x):
        raise NotImplementedError("This needs to be implemented by"
                                  "your Policy class.")

    def get_playable_predictions(self, x):
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

        playable_bool = self.get_playable_bool(x)
        # Get the Q value estimations from the estimator.
        q = self.estimator.predict(x)[0]
        # Filter so that not playable cards have the min Q values.
        q_playable = q.copy()
        q_playable[~playable_bool] = np.min(q) - 1
        return q_playable

    def get_playable_bool(self, x):
        """
        Returns a boolean array of the playable actions.
        Args:
            x:  Game state, the first 54 elements should describe the hand
            of the player.

        Returns:
            playable_bool: boolean array of playable actions.

        """
        playable = x[:Card.DIFFERENT_CARDS]
        # A player can have 0-4 Z(53/-1) or N(52/-2) cards in their hand.
        # We need to make this into a "playable/not-playable" bool array.
        # Anything above a 1 becomes 1(playable).
        if playable[-2] >= 1:
            playable[-2] = 1
        if playable[-1] >= 1:
            playable[-1] = 1
        playable_bool = np.array(playable).astype(bool)
        return playable_bool


class MaxPolicy(Policy):

    def __init__(self, estimator):
        assert isinstance(estimator, PolicyGradient)
        self.logger = logging.getLogger('wizard-rl.MaxPolicy')

        super().__init__(estimator)

    def get_action(self, x):
        q_playable = self.get_playable_predictions(x)
        a = np.argmax(q_playable)
        return a


class EGreedyPolicy(Policy):

    def __init__(self, estimator, epsilon, decay=0.99):
        super().__init__(estimator)
        self.logger = logging.getLogger('wizard-rl.EGreedyPolicy')

        self.epsilon = epsilon
        self.decay_rate = decay
        # In case we need to reset epsilon.
        self.original_epsilon = epsilon
        self.curr_epsilon = 1 - epsilon

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
        self.curr_epsilon *= self.decay_rate
        epsilon = self.curr_epsilon + self.epsilon
        num_a = Card.DIFFERENT_CARDS
        playable_bool = self.get_playable_bool(x)
        q_playable = self.get_playable_predictions(x)

        # All probabilities start as 0
        probs = np.zeros(num_a)
        # Only potential actions are the playable ones.
        # assign epsilon probabilities to every potential action.
        probs[playable_bool] += epsilon/sum(playable_bool)
        # self.logger.info(q_playable)
        # Find the greedy action
        greedy_a = np.argmax(q_playable)
        # Give it the highest probability.
        probs[greedy_a] += (1-epsilon)
        # self.logger.info(probs)
        a = np.random.choice(len(probs), p=probs)
        return a
