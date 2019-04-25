from RLAgents import RLAgent
from Player import AverageRandomPlayer
from Wizard import Wizard
from Estimators.ValueEstimators import DQNEstimator, DoubleDQNEstimator, DuelingDQNEstimator
from Featurizers import Featurizer
from TrickPrediction import TrickPrediction
from Policies import MaxPolicy
from Estimators.PolicyEstimators import PolicyGradient

from plotting import plot_moving_average_wins

import time
import numpy as np
import tensorflow as tf


class WizardStatistic(object):

    def __init__(self, players, num_games=20):
        self.num_games = num_games
        self.num_players = len(players)
        self.wins = np.zeros((num_games, len(players)))
        self.scores = np.zeros((num_games, len(players)))
        self.players = players

    def play_games(self):
        for i in range(self.num_games):
            wiz = Wizard(num_players=self.num_players, players=self.players)
            scores = wiz.play()
            # evaluate scores
            self.wins[i][scores == np.max(scores)] = 1
            self.scores[i] = scores
            print("{0}: {1}".format(i, scores))
            print("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

    def plot_game_statistics(self, interval=200):
        path_name = 'log/statistics/' + time.strftime("%Y-%m-%d_%H-%M-%S")

        plot_moving_average_wins(self.players, self.wins, self.scores, path_name, interval=interval)


if __name__ == "__main__":
    tf.reset_default_graph()

    with tf.Session() as sess:
        featurizer = Featurizer()
        dueling_dqn = DuelingDQNEstimator(sess, input_shape=featurizer.get_state_size())
        dqn_estimator = DQNEstimator(sess, input_shape=featurizer.get_state_size())
        double_estimator = DoubleDQNEstimator(sess, input_shape=featurizer.get_state_size())
        tp = TrickPrediction(sess)
        pg_estimator = PolicyGradient(sess, input_shape=featurizer.get_state_size())
        max_policy = MaxPolicy(pg_estimator)
        dueling_agent = RLAgent(featurizer=featurizer, estimator=dueling_dqn)
        dqn_agent = RLAgent(featurizer=featurizer, estimator=dqn_estimator)
        ddqn_agent = RLAgent(featurizer=featurizer, estimator=double_estimator)
        pg_agent = RLAgent(featurizer=featurizer, estimator=pg_estimator, policy=max_policy)
        sess.run(tf.global_variables_initializer())

        players = [AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   dueling_agent]

        stat = WizardStatistic(players, num_games=1000)
        stat.play_games()
        stat.plot_game_statistics(interval=200)
