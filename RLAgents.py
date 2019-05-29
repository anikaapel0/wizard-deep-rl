from Estimators import ValueEstimators
from Estimators.PolicyEstimators import PolicyGradient
import Policies
import Featurizers
from Player import AverageRandomPlayer

import logging


class RLAgent(AverageRandomPlayer):
    """A computer player that learns using reinforcement learning."""

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None):
        super().__init__()
        self.logger = logging.getLogger('wizard-rl.RLAgent')
        if featurizer is None:
            self.featurizer = Featurizers.Featurizer()
        else:
            self.featurizer = featurizer
        if estimator is None:
            assert session is not None
            self.estimator = ValueEstimators.DQNEstimator(session, input_shape=self.featurizer.get_state_size())
        else:
            self.estimator = estimator
        if policy is None:
            self.policy = Policies.EGreedyPolicy(self.estimator, epsilon=0.1)
        else:
            self.policy = policy
        self.trick_prediction = trick_prediction

        self.old_state = None
        self.old_score = 0
        self.old_action = None
        self.whole_hand = None
        self.training_mode = True

    def get_prediction(self, trump, predictions, players, restriction=None):
        if self.trick_prediction is None:
            return super(RLAgent, self).get_prediction(trump, predictions, players, restriction)

        s = self.featurizer.transform_handcards(self, trump)
        average = len(self.whole_hand) // len(players)
        prediction = self.trick_prediction.predict(s, average)

        # round prediction
        final_pred = int(round(prediction))
        if restriction is not None and final_pred == restriction:
            if prediction < final_pred:
                final_pred -= 1
            else:
                final_pred += 1
        # self.logger.info("Prediction: {}, Hand: {}, Trumpf: {}".format(final_pred, self.whole_hand, trump))

        return final_pred

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

    def enable_training(self):
        self.training_mode = True

    def disable_training(self):
        self.training_mode = False

    def save_estimator(self, name="default"):
        self.estimator.save(name)

    def load_estimator(self, name="default"):
        self.estimator.load(name)

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

    def trick_ended(self, trump):
        if self.training_mode:
            if self.trick_prediction is not None:
                arr_cards = self.featurizer.transform_handcards(self, trump)
                self.trick_prediction.update(arr_cards, self.prediction, self.wins)

            self.policy.decay_epsilon()

    def name(self):
        return self.estimator.name()


class DQNAgent(RLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None):
        super(DQNAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer, trick_prediction=trick_prediction, session=session)


class DoubleDQNAgent(RLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()

        if estimator is None:
            assert session is not None
            estimator = ValueEstimators.DoubleDQNEstimator(session, input_shape=featurizer.get_state_size())
        super(DoubleDQNAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer, trick_prediction=trick_prediction, session=session)


class PGAgent(RLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()
        if policy is None:
            policy = Policies.MaxPolicy(featurizer)
        if estimator is None:
            assert session is not None
            estimator = PolicyGradient(session, featurizer.get_state_size())
        else:
            assert isinstance(estimator, PolicyGradient)

        super(PGAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer, trick_prediction=trick_prediction, session=session)
