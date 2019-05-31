import logging
from random import shuffle, choice
from collections import Counter

from Environment import Card
from GamePlayer import Featurizers
from GamePlayer import Policies
from GamePlayer.Estimators import ValueEstimators, PolicyEstimators


class CardPlayer(object):

    def __init__(self):
        self.logger = logging.getLogger('wizard-rl.CardPlayer')
        self.hand = []
        self.score = 0
        self.reward = 0
        self.wins = 0
        self.whole_hand = None
        super(CardPlayer, self).__init__()

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

    def get_trump_color(self):
        return choice(Card.Card.colors[1:])


class RandomCardPlayer(CardPlayer):

    def __init__(self):
        super(RandomCardPlayer, self).__init__()

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


class RLCardPlayer(CardPlayer):

    def __init__(self):
        self.logger = logging.getLogger('wizard-rl.RLAgent')
        self.session = None
        self.path = None
        self.featurizer = None
        self.estimator = None
        self.policy = None
        self.old_state = None
        self.old_score = 0
        self.old_action = None
        self.whole_hand = None
        self.training_mode = True
        super(RLCardPlayer, self).__init__()

    def init_player(self, session, path, estimator=None, policy=None, featurizer=None):
        self.session = session
        self.path = path
        if featurizer is None:
            self.featurizer = Featurizers.Featurizer()
        else:
            self.featurizer = featurizer
        if estimator is None:
            assert session is not None
            self.estimator = ValueEstimators.DQNEstimator(session, input_shape=self.featurizer.get_state_size(), path=path)
        else:
            self.estimator = estimator
        if policy is None:
            self.policy = Policies.EGreedyPolicy(self.estimator, epsilon=0.1)
        else:
            self.policy = policy

    def play_card(self, trump, first, played, players, played_in_game):
        """Plays a card according to the estimator Q function and learns
        on-line.
        Relies on scores being updated by the environment to calculate reward.
        Args:
            trump: (Card) trump card.
            first: (Card) first card.
            played: (list(Card)) list of cards played in Trick, may be empty.
            players: (list(Player)) list of players in the game, including this
            player.
            played_in_game: (list(Card)) list of cards played so far in the
            game, may be empty.

        Returns:
            card_to_play: (Card) the card object that the player
             decided to play.
        """
        state = self.featurizer.transform(self, trump, first, played, players,
                                          played_in_game)
        terminal = False
        if self.old_state is not None and self.old_action is not None and self.training_mode:
            r = self.reward
            if r != 0:
                terminal = True
                # If we got a reward, it's a terminal state.
                # We signal this with an s_prime == None
                self.estimator.update(self.old_state, self.old_action, r, None)
            else:
                self.estimator.update(self.old_state, self.old_action, r, state)

        a = self.policy.get_action(state)
        card_to_play = self._remove_card_played(a)
        self.old_state = None if terminal else state
        self.old_action = a
        self.give_reward(0)  # After playing a card, the reward is 0.
        # Unless it's the last card of the game, then the Game object will
        # call give_reward before the next play_card, setting the correct reward
        return card_to_play

    def _remove_card_played(self, a):
        """
        Given an action (integer) remove a card equivalent to it from the
        player's hand and return it.

        Args:
            a: (int) The action taken. Remove a card with the same code.
            If there is more than one that matches, it does not matter which,
            but just remove one.

        Returns:
            card_to_play: The card corresponding to the action.

        Raises:
            RuntimeError when the action does not correspond to any card.

        """
        card_to_return = None
        for card in self.hand:
            if int(card) == a:
                card_to_return = card
                self.hand.remove(card)
                break
        if card_to_return is None:
            raise RuntimeError("Computer did not find a valid card for this"
                               "action.\nHand: {}\nAction: {}".format(self.hand,
                                                                      a))
        return card_to_return

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





