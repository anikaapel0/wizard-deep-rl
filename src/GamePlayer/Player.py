from random import choice
from collections import Counter
import logging

from GamePlayer.PredictionPlayer import PredictionPlayer, RandomPredictionPlayer, AveragePredictionPlayer, NetworkPredictionPlayer
from GamePlayer.CardPlayer import CardPlayer, RandomCardPlayer, RLCardPlayer
from Environment import Card


class Player(object):

    def __init__(self):
        self.logger = logging.getLogger('wizard-rl.Player')
        self.hand = []
        self.score = 0
        self.reward = 0
        self.wins = 0
        self.whole_hand = None
        super(Player, self).__init__()

    # def get_prediction(self, trump, predictions, players, game_num, restriction=None):
    #     raise NotImplementedError("This needs to be implemented by your Player class")

    # def play_card(self, trump, first, played, players, played_in_game):
    #     raise NotImplementedError("This needs to be implemented by your Player class")

    # def get_trump_color(self):
    #     raise NotImplementedError("This needs to be implemented by your Player class")

    def trick_ended(self, trump):
        return

    def give_reward(self, reward):
        self.reward = reward
        self.score += reward

    def get_state(self):
        return self.score, self.wins, self.prediction

    def reset_score(self):
        self.score = 0

    def name(self):
        raise NotImplementedError("This needs to be implemented by your Player class")


class RandomPlayer(Player, RandomCardPlayer, RandomPredictionPlayer):
    """A completely random agent, it always chooses all
    its actions randomly"""

    def __init__(self):
        super().__init__()

    def name(self):
        return "RandomPlayer"


class AverageRandomPlayer(Player, RandomCardPlayer, AveragePredictionPlayer):
    """Agent that uses random cards, but chooses an 'average'
    prediction of wins and a trump color corresponding to
    the color the agent has the most of in its hand."""

    def __init__(self):
        super().__init__()

    def get_trump_color(self):
        # Return the color the agent has the most of in its hand.
        color_counter = Counter()
        for card in self.hand:
            color = card.color
            if color == "White":
                continue
            color_counter[color] += 1
        if not color_counter.most_common(1):
            return super().get_trump_color()
        else:
            return color_counter.most_common(1)[0][0]

    def name(self):
        return "AverageRandomPlayer"


class PredictionRandomPlayer(Player, RandomCardPlayer, NetworkPredictionPlayer):

    def __init__(self, session, trick_prediction=None, featurizer=None):
        super().__init__()
        self.init_prediction(session, trick_prediction, featurizer)

    def trick_ended(self, trump):
        arr_cards = self.featurizer.transform_handcards(self, trump)
        self.trick_prediction.update(arr_cards, self.prediction, self.wins)

    def name(self):
        return "PredictionRandomPlayer"
