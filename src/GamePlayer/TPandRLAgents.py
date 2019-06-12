from GamePlayer.Estimators import ValueEstimators
from GamePlayer.Estimators.PolicyEstimators import PolicyGradient
from GamePlayer import Policies, Featurizers
from GamePlayer.Player import AverageRandomPlayer, Player
from GamePlayer.CardPlayer import RLCardPlayer
from GamePlayer.PredictionPlayer import NetworkPredictionPlayer, AveragePredictionPlayer
from GamePlayer.TrickPrediction import TrickPrediction

import logging


class TPRLAgent(Player, RLCardPlayer, NetworkPredictionPlayer):
    """A computer player that learns using reinforcement learning."""

    def __init__(self, session, path, estimator=None, policy=None, featurizer=None, trick_prediction=None):
        self.logger = logging.getLogger('wizard-rl.TPRLAgent')
        if trick_prediction is None:
            trick_prediction = TrickPrediction(session, path)
        super().__init__()
        self.init_player(session, path, estimator=estimator, policy=policy, featurizer=featurizer)
        self.init_prediction(session, trick_prediction, featurizer=featurizer)

    def enable_training(self):
        self.training_mode = True

    def disable_training(self):
        self.training_mode = False

    def save_estimator(self, name="default"):
        self.estimator.save(name)

    def load_estimator(self, name="default"):
        self.estimator.load(name)

    def trick_ended(self, trump):
        if self.training_mode:
            if self.trick_prediction is not None:
                arr_cards = self.featurizer.transform_handcards(self, trump)
                self.trick_prediction.update(arr_cards, self.prediction, self.wins)

            self.policy.decay_epsilon()

    def name(self):
        return self.estimator.name() + " TP"


class TPDQNAgent(TPRLAgent):

    def __init__(self, session, path, estimator=None, policy=None, featurizer=None, trick_prediction=None):
        super(TPDQNAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                         trick_prediction=trick_prediction, session=session, path=path)


class TPDoubleDQNAgent(TPRLAgent):

    def __init__(self, session, path, estimator=None, policy=None, featurizer=None, trick_prediction=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()

        if estimator is None:
            assert session is not None
            estimator = ValueEstimators.DoubleDQNEstimator(session, input_shape=featurizer.get_state_size(), path=path)
        super(TPDoubleDQNAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                               trick_prediction=trick_prediction, session=session, path=path)


class TPDuelingAgent(TPRLAgent):

    def __init__(self, session, path, estimator=None, policy=None, featurizer=None, trick_prediction=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()

        if estimator is None:
            assert session is not None
            estimator = ValueEstimators.DuelingDQNEstimator(session, input_shape=featurizer.get_state_size(), path=path)
        super(TPDuelingAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                             trick_prediction=trick_prediction, session=session, path=path)


class TPPGAgent(TPRLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None, path=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()
        if estimator is None:
            assert session is not None
            estimator = PolicyGradient(session, featurizer.get_state_size(), path=path)
        else:
            assert isinstance(estimator, PolicyGradient)
        if policy is None:
            policy = Policies.MaxPolicy(estimator)

        super(TPPGAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                        trick_prediction=trick_prediction, session=session, path=path)
