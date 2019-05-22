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

    def __init__(self, session, players, num_games=20, interval=500):
        self.logger = logging.getLogger('wizard-rl.WizardStatistics.WizardStatistic')
        self.num_games = num_games
        self.interval = interval
        self.num_players = len(players)
        self.wins = np.zeros((num_games, len(players)))
        self.scores = np.zeros((num_games, len(players)))
        self.players = players
        self.score_writer = None
        self.score_window = None
        self.session = session
        self.wins_window = None
        self._merged = None
        self._init_tracking()

    def _init_tracking(self):
        self.score_window = tf.placeholder("float", [None, self.num_players])
        self.wins_window = tf.placeholder("float", [None, self.num_players])
        self.curr_scores = tf.reduce_sum(self.score_window, axis=0) / self.interval
        self.curr_wins = tf.reduce_sum(self.wins_window, axis=0) / self.interval
        merging = []
        for i in range(self.num_players):
            if isinstance(self.players[i], RLAgent):
                name = self.players[i].estimator.name()
            else:
                name = "RandomPlayer"

            merging.append(tf.summary.scalar('curr_score_{}_{}'.format(i, name), self.curr_scores[i]))
            merging.append(tf.summary.scalar('curr_wins_{}_{}'.format(i, name), self.curr_wins[i]))

        self._merged = tf.summary.merge(merging)
        self.score_writer = tf.summary.FileWriter("log/tf_statistics", self.session.graph)

    def play_games(self):
        for i in range(self.num_games):
            wiz = Wizard(num_players=self.num_players, players=self.players)
            scores = wiz.play()
            # evaluate scores
            self.wins[i][scores == np.max(scores)] = 1
            self.scores[i] = scores
            if i > self.interval:
                curr_scores = self.scores[i - self.interval:i]
                curr_wins = self.wins[i - self.interval: i]
                summary, _, _ = self.session.run([self._merged, self.curr_wins, self.curr_scores],
                                                         feed_dict={self.score_window: curr_scores,
                                                                    self.wins_window: curr_wins})

                self.score_writer.add_summary(summary, i)
            self.logger.info("{0}: {1}".format(i, scores))
            self.logger.info("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

    def plot_game_statistics(self):
        path = "log/statistics/"
        if not os.path.exists(path):
            os.makedirs(path)
        filename = time.strftime("%Y-%m-%d_%H-%M-%S")

        plot_moving_average_wins(self.players, self.wins, self.scores, path + filename, interval=self.interval)

    def get_winner(self):
        last_wins = np.sum(self.wins[-1000:], axis=0)

        return self.players[np.argmax(last_wins)]


def init_logger(console_logging=False, console_level=logging.ERROR):
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
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


if __name__ == "__main__":
    tf.reset_default_graph()
    init_logger(console_logging=True, console_level=logging.DEBUG)

    with tf.Session() as sess:
        featurizer = Featurizer()
        dueling_dqn = DuelingDQNEstimator(sess, input_shape=featurizer.get_state_size())
        dqn_estimator = DQNEstimator(sess, input_shape=featurizer.get_state_size())
        double_estimator = DoubleDQNEstimator(sess, input_shape=featurizer.get_state_size())
        tp = TrickPrediction(sess)
        pg_estimator = PolicyGradient(sess, input_shape=featurizer.get_state_size())
        max_policy = MaxPolicy(pg_estimator)
        dueling_agent = RLAgent(featurizer=featurizer, estimator=dueling_dqn, trick_prediction=tp)
        dqn_agent = RLAgent(featurizer=featurizer, estimator=dqn_estimator, trick_prediction=tp)
        ddqn_agent = RLAgent(featurizer=featurizer, estimator=double_estimator, trick_prediction=tp)
        pg_agent = RLAgent(featurizer=featurizer, estimator=pg_estimator, policy=max_policy, trick_prediction=tp)

        players = [AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   dqn_agent]

        stat = WizardStatistic(sess, players, num_games=10000, interval=500)
        sess.run(tf.global_variables_initializer())

        stat.play_games()
        stat.plot_game_statistics()
