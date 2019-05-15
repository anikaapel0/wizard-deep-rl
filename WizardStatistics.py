from RLAgents import RLAgent
from Player import AverageRandomPlayer, PredictionRandomPlayer
from GameUtilities.Wizard import Wizard
from Estimators.ValueEstimators import DQNEstimator, DoubleDQNEstimator, DuelingDQNEstimator
from Featurizers import Featurizer
from TrickPrediction import TrickPrediction
from Policies import MaxPolicy
from Estimators.PolicyEstimators import PolicyGradient

from plotting import plot_moving_average_wins

import logging
import time
import os
import numpy as np
import tensorflow as tf


class WizardStatistic(object):

    def __init__(self, players, num_games=20):
        self.logger = logging.getLogger('wizard-rl.WizardStatistics.WizardStatistic')
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
            self.logger.info("{0}: {1}".format(i, scores))
            self.logger.info("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

    def plot_game_statistics(self, interval=200):
        path = "log/statistics/"
        if not os.path.exists(path):
            os.makedirs(path)
        filename = time.strftime("%Y-%m-%d_%H-%M-%S")

        plot_moving_average_wins(self.players, self.wins, self.scores, path + filename, interval=interval)


def init_logger(console_logging=False):
    # create logger with 'wizard-rl'
    logger = logging.getLogger('wizard-rl')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs debug messages
    path = "log/"
    filename = path + "wizard-rl.log"
    if not os.path.exists(path):
        os.makedirs(path)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(fh)

    if console_logging:
        # create console handler with higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


if __name__ == "__main__":
    tf.reset_default_graph()
    init_logger(console_logging=True)

    with tf.Session() as sess:
        featurizer = Featurizer()
        dueling_dqn = DuelingDQNEstimator(sess, input_shape=featurizer.get_state_size())
        dqn_estimator = DQNEstimator(sess, input_shape=featurizer.get_state_size())
        double_estimator = DoubleDQNEstimator(sess, input_shape=featurizer.get_state_size())
        tp = TrickPrediction(sess)
        pg_estimator = PolicyGradient(sess, input_shape=featurizer.get_state_size())
        max_policy = MaxPolicy(pg_estimator)
        dueling_agent = RLAgent(featurizer=featurizer, estimator=dueling_dqn)
        dqn_agent = RLAgent(featurizer=featurizer, estimator=dqn_estimator, trick_prediction=tp)
        ddqn_agent = RLAgent(featurizer=featurizer, estimator=double_estimator)
        pg_agent = RLAgent(featurizer=featurizer, estimator=pg_estimator, policy=max_policy)
        sess.run(tf.global_variables_initializer())

        players = [PredictionRandomPlayer(sess, tp, featurizer),
                   PredictionRandomPlayer(sess, tp, featurizer),
                   PredictionRandomPlayer(sess, tp, featurizer),
                   dqn_agent]

        stat = WizardStatistic(players, num_games=10000)
        stat.play_games()
        stat.plot_game_statistics(interval=500)
