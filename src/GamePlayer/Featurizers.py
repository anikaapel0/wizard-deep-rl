from Environment import Card
import random
from copy import deepcopy
import logging


class Featurizer(object):
    """This class takes a state (all data given to choose a card to play)
    and transforms it into an array that is useful for an estimator."""

    def __init__(self, count_cards=True):
        self.logger = logging.getLogger('wizard-rl.Featurizer')

        self.count_cards = count_cards

    def transform_handcards(self, player, trump):
        hand_arr = self.cards_to_arr(player.whole_hand)
        trump_color = self.color_to_bin_arr(trump)

        return hand_arr + trump_color

    def transform(self, player, trump, first, played, players, played_in_game):
        """
        Transforms the state into a state compatible with estimators.

        Args:
            player: (Player) the player who is calling the featurizer.
            trump: (Card) trump card.
            first: (Card) first card.
            played: (list(Card)) list of played cards in the trick, may be
            empty.
            players: (list(Players)) list of players in the game, includes
            THIS player.
            played_in_game: (list(Cards)) list of cards played in the game,
            may be empty.

        Returns:
            state: The state, compatible with the estimator.
        """
        hand_arr = self.cards_to_arr(player.hand)
        trick_arr = self.cards_to_arr(played)
        if self.count_cards:
            game_arr = self.cards_to_arr(played_in_game)
        else:
            game_arr = []
        trump_color = self.color_to_bin_arr(trump)
        first_color = self.color_to_bin_arr(first)
        player_score_win_predict = self.players_to_arr(players, player)

        return hand_arr + trick_arr + game_arr + trump_color + first_color \
               + player_score_win_predict

    @staticmethod
    def cards_to_arr(cards):
        """
        Transforms cards into an array. All cards are binary (either in the
        list or not) except Z and N which are between 0 or 4. Indices are
        given by the Card.__int__ method. int(Z) == 53, int(N) == 52
        Args:
            cards: (list(Card)) list of cards to transform into array.

        Returns:
            arr: array (len==54) indicating the count of each card.
        """
        arr = [0] * Card.Card.DIFFERENT_CARDS
        for card in cards:
            arr[int(card)] += 1
        return arr

    @staticmethod
    def cards_to_arr_trump_first(cards, trump):
        arr = [0] * Card.Card.DIFFERENT_CARDS

        colors = deepcopy(Card.Card.color_order)
        if trump.color == "White":
            random.shuffle(colors)
            shuffled_colors = colors
        else:
            colors.remove(trump.color)
            random.shuffle(colors)
            shuffled_colors = [trump.color] + colors

        for card in cards:
            arr[card.card_to_int_trump(shuffled_colors)] += 1
        return arr

    @staticmethod
    def color_to_bin_arr(card):
        """
        Transforms a color into a one-hot encoding of it. The index order is
        given by Card.Card.colors .
        Args:
            card: (Card) the card to extract color from.
            May be none.

        Returns:
            arr: one-hot encoding array of the color

        """
        bin_arr = [0] * len(Card.Card.colors)
        if card is None:
            return bin_arr
        else:
            color = card.color
        index = Card.Card.colors.index(color)
        bin_arr[index] = 1
        return bin_arr

    @staticmethod
    def players_to_arr(players, player):
        """
        Returns an array of the form [win1, predict1, ...,
        wini, predicti, winPersonal, predictPersonal]
        With the wins and predictions of each player finally with the
        wins and predictions of THIS player (player).
        -1 for any "non existent player".

        Args:
            players: list of all players
            player: THIS player

        Returns:
            arr: a list with the scores, wins, predictions of all players.
        """
        arr = []
        for other_player in players:
            if other_player == player:
                continue
            state = list(other_player.get_state())
            arr += state[1:]
        state = list(player.get_state())
        arr += state[1:]
        return arr

    def get_state_size(self):
        count = 0
        # played cards
        if self.count_cards:
            count += 54

        count += 126

        return count


def default_featurizer_state_size():
    return Featurizer().get_state_size()
