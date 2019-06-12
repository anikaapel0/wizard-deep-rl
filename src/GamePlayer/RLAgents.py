from GamePlayer.Estimators import ValueEstimators
from GamePlayer.Estimators.PolicyEstimators import PolicyGradient
from GamePlayer import Policies, Featurizers
from GamePlayer.Player import AverageRandomPlayer, Player
from GamePlayer.CardPlayer import RLCardPlayer
from GamePlayer.PredictionPlayer import AveragePredictionPlayer

import logging


class RLAgent(Player, RLCardPlayer, AveragePredictionPlayer):
    """A computer player that learns using reinforcement learning."""

    def __init__(self, session, path, estimator=None, policy=None, featurizer=None):
        self.logger = logging.getLogger('wizard-rl.RLAgent')
        super().__init__()
        self.init_player(session, path, estimator=estimator, policy=policy, featurizer=featurizer)

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

            self.policy.decay_epsilon()

    def name(self):
        return self.estimator.name()


class DQNAgent(RLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None, path=None):
        super(DQNAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                       trick_prediction=trick_prediction, session=session, path=path)


class DoubleDQNAgent(RLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None, path=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()

        if estimator is None:
            assert session is not None
            estimator = ValueEstimators.DoubleDQNEstimator(session, input_shape=featurizer.get_state_size(), path=path)
        super(DoubleDQNAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                             trick_prediction=trick_prediction, session=session, path=path)


class DuelingAgent(RLAgent):

    def __init__(self, estimator=None, policy=None, featurizer=None, trick_prediction=None, session=None, path=None):
        if featurizer is None:
            featurizer = Featurizers.Featurizer()

        if estimator is None:
            assert session is not None
            estimator = ValueEstimators.DuelingDQNEstimator(session, input_shape=featurizer.get_state_size(), path=path)
        super(DuelingAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                           trick_prediction=trick_prediction, session=session, path=path)


class PGAgent(RLAgent):

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

        super(PGAgent, self).__init__(estimator=estimator, policy=policy, featurizer=featurizer,
                                      trick_prediction=trick_prediction, session=session, path=path)
