from random import shuffle, randrange, choice, random
from collections import Counter
import Card
import Featurizers

import logging


class Player(object):

    def __init__(self):
        self.hand = []
        self.score = 0
        self.reward = 0
        self.wins = 0
        self.prediction = -1
        self.whole_hand = None
        self.trick_prediction = None

    def get_playable_cards(self, first):
        playable_cards = []
        first_colors = []
        if first is None:
            return self.hand
        for card in self.hand:
            # White cards can ALWAYS be played.
            if card.color == "White":
                playable_cards.append(card)
            # First card color can ALWAYS be played.
            elif card.color == first.color:
                first_colors.append(card)
            # Other colors can only be played if there
            # no cards of the first color in the hand.
        if len(first_colors) > 0:
            return playable_cards + first_colors
        else:
            # Cannot follow suit, use ANY card.
            return self.hand

    def play_card(self, trump, first, played, players, played_in_game):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def get_prediction(self, trump, predictions, players, restriction=None):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def get_trump_color(self):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def trick_ended(self, trump):
        if self.trick_prediction is not None:
            arr_cards = self.featurizer.transform_handcards(self, trump)
            self.trick_prediction.update(arr_cards, self.prediction, self.wins)

    def give_reward(self, reward):
        self.reward = reward
        self.score += reward

    def get_state(self):
        return self.score, self.wins, self.prediction

    def reset_score(self):
        self.score = 0

    def close(self):
        return


class RandomPlayer(Player):
    """A completely random agent, it always chooses all
    its actions randomly"""

    def __init__(self):
        super().__init__()

    def play_card(self, trump, first, played, players, played_in_game):
        """Randomly play any VALID card.
        Returns:
            card_to_play: (Card) the chosen card from the player hand.
            """
        possible_actions = super().get_playable_cards(first)
        if not isinstance(possible_actions, list):
            possible_actions = list(possible_actions)
        shuffle(possible_actions)
        card_to_play = possible_actions[0]
        self.hand.remove(card_to_play)
        # self.logger.info("Playing card {} from {}".format(card_to_play, self.hand))
        return card_to_play

    def get_prediction(self, trump, predictions, players, restriction=None):
        """Randomly return any number of wins between 0 and total number
         of games.
         """
        prediction = randrange(len(self.hand))
        if prediction == restriction:
            if random():
                prediction += 1
            else:
                prediction -= 1

        self.prediction = prediction
        return prediction

    def get_trump_color(self):
        # Randomly return any color except white.
        return choice(Card.Card.colors[1:])


class AverageRandomPlayer(RandomPlayer):
    """Agent that uses random cards, but chooses an 'average'
    prediction of wins and a trump color corresponding to
    the color the agent has the most of in its hand."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('wizard-rl.Player.AverageRandomPlayer')

    def get_prediction(self, trump, predictions, players, restriction=None):
        prediction = len(self.hand) // len(predictions)
        if prediction == restriction:
            if random():
                prediction += 1
            else:
                prediction -= 1
        self.prediction = prediction
        return prediction

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


class TrickPredictionRandomPlayer(RandomPlayer):
    def __init__(self, trick_prediction, featurizer=None):
        super().__init__()
        self.logger = logging.getLogger('wizard-rl.Player.TrickPredictionRandomPlayer')
        self.trick_prediction = trick_prediction
        if featurizer is None:
            self.featurizer = Featurizers.Featurizer()
        else:
            self.featurizer = featurizer

    def get_prediction(self, trump, predictions, players, restriction=None):
        s = self.featurizer.transform_handcards(self, trump)
        average = len(self.hand) // len(predictions)
        prediction = self.trick_prediction.predict(s, average)

        # round prediction
        final_pred = int(round(prediction))
        self.logger.info("Prediction: {}, Hand: {}, Trump: {}".format(final_pred, self.whole_hand, trump))
        if restriction is not None and final_pred == restriction:
            if prediction < final_pred:
                final_pred -= 1
            else:
                final_pred += 1

        return final_pred

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


class FunctionRandomPlayer(RandomPlayer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('wizard-rl.Player.FunctionRandomPlayer')

    def get_prediction(self, trump, predictions, players, restriction=None):
        score = 0
        # loop through cards
        for card in self.hand:
            # zauberer
            if card.is_z():
                score += 0.99
            # trump card
            elif card.color == trump.color:
                score += (0.9/12 * card.value - 0.075)
            # normal card
            else:
                score += (0.8/6 * card.value - 28/30)

        final_score = round(score)

        if restriction is not None and final_score == restriction:
            if score < final_score:
                final_score -= 1
            else:
                final_score += 1

        return final_score

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
