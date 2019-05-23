from RLAgents import RLAgent
from Player import AverageRandomPlayer, PredictionRandomPlayer
from GameUtilities.Wizard import Wizard
from GameUtilities.Game import Game
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


class WizardTraining(object):

    def __init__(self, session, players, num_games=20, interval=500,
                 path="log/tf_statistics", evaluation_games=1000):
        self.logger = logging.getLogger('wizard-rl.WizardStatistics.WizardStatistic')
        self.num_games = num_games
        self.interval = interval
        self.num_players = len(players)
        self.evaluation_games = evaluation_games
        self.wins = np.zeros((num_games, len(players)))
        self.scores = np.zeros((num_games, len(players)))
        self.players = players
        self.path = path
        self.score_writer = None
        self.score_window = None
        self.session = session
        self.wins_window = None
        self.t_eval = 0
        self._merged = None
        self._init_tracking()

    def _init_tracking(self):
        self.score_window = tf.placeholder("float", [None, self.num_players])
        self.wins_window = tf.placeholder("float", [None, self.num_players])
        self.curr_scores = tf.reduce_sum(self.score_window, axis=0) / self.evaluation_games
        self.curr_wins = tf.reduce_sum(self.wins_window, axis=0) / self.evaluation_games
        merging = []
        for i in range(self.num_players):
            if isinstance(self.players[i], RLAgent):
                name = self.players[i].estimator.name()
            else:
                name = "RandomPlayer"

            merging.append(tf.summary.scalar('curr_score_{}_{}'.format(i, name), self.curr_scores[i]))
            merging.append(tf.summary.scalar('curr_wins_{}_{}'.format(i, name), self.curr_wins[i]))

        self._merged = tf.summary.merge(merging)
        self.score_writer = tf.summary.FileWriter(self.path, self.session.graph)

    def update_scores(self, scores, wins):
        summary, mean_scores, mean_wins = self.session.run([self._merged, self.curr_wins, self.curr_scores],
                                                           feed_dict={self.score_window: scores,
                                                                      self.wins_window: wins})

        self.score_writer.add_summary(summary, self.t_eval)
        self.logger.info("Evaluation {0}\n\tScores: {1}\n\tWins: {2}".format(self.t_eval, mean_scores, mean_wins))

    def play_same_round(self, num_cards):
        for i in range(self.num_games):
            wiz_round = Game(num_cards, self.players, i % 4)
            scores = wiz_round.play()

            self.update_scores(scores, i)

    def disable_training(self):
        for player in self.players:
            if isinstance(player, RLAgent):
                player.disable_training()

    def enable_training(self):
        for player in self.players:
            if isinstance(player, RLAgent):
                player.enable_training()

    def train_agents(self, random_agents_interval):
        for i in range(self.num_games):
            if i % 100 == 0:
                self.logger.info("Playing training round {}".format(i))

            wiz = Wizard(num_players=self.num_players, players=self.players)
            wiz.play()

            if i % random_agents_interval == 0:
                self.play_evaluation_games()

    def play_evaluation_games(self):
        self.disable_training()
        self.t_eval += 1
        self.logger.info("Starting a train evaluation game with random players")
        players = [AverageRandomPlayer() for _ in range(len(self.players) - 1)]
        players.append(self.players[0])

        # arrays for evaluation scores and
        eval_scores = np.zeros((self.evaluation_games, len(players)))
        eval_wins = np.zeros((self.evaluation_games, len(players)))

        for i in range(self.evaluation_games):
            wiz = Wizard(num_players=len(players), players=players)
            scores = wiz.play()
            # evaluate scores
            eval_scores[i] = scores
            eval_wins[i][scores == np.max(scores)] = 1

        self.update_scores(eval_scores, eval_wins)
        self.enable_training()

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
        path = "log/start_" + time.strftime("%Y-%m-%d_%H-%M-%S")

        dueling_dqn = DuelingDQNEstimator(sess, input_shape=featurizer.get_state_size(), path=path)
        # dqn_estimator = DQNEstimator(sess, input_shape=featurizer.get_state_size(), path=path)
        # double_estimator = DoubleDQNEstimator(sess, input_shape=featurizer.get_state_size(), path=path)
        tp = TrickPrediction(sess)
        # pg_estimator = PolicyGradient(sess, input_shape=featurizer.get_state_size(), path=path)
        # max_policy = MaxPolicy(pg_estimator)
        dueling_agent = RLAgent(featurizer=featurizer, estimator=dueling_dqn, trick_prediction=tp)
        # dqn_agent = RLAgent(featurizer=featurizer, estimator=dqn_estimator, trick_prediction=tp)
        # ddqn_agent = RLAgent(featurizer=featurizer, estimator=double_estimator, trick_prediction=tp)
        # pg_agent = RLAgent(featurizer=featurizer, estimator=pg_estimator, policy=max_policy, trick_prediction=tp)

        players = [AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   dueling_agent]

        training = WizardTraining(sess, players, num_games=7000, interval=500, path=path)
        sess.run(tf.global_variables_initializer())

        training.train_agents(500)
        # stat.play_same_round(10)
        # training.plot_game_statistics()
