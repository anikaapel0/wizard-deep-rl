import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Estimator.ValueEstimators import DQNEstimator
from Estimator.PolicyEstimator import PolicyGradient
from Featurizers import Featurizer
from NNTrickPrediction import NNTrickPrediction
from Player import RandomPlayer, AverageRandomPlayer, TrickPredictionRandomPlayer
from RLAgents import RLAgent
from Wizard import Wizard


class WizardStatistic(object):

    plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']

    def __init__(self, num_games=20, players=None, num_players=4, num_agents=1, trick_prediction=False):
        self.logger = logging.getLogger('wizard-rl.WizardStatistics.WizardStatistic')
        self.logger.info('starting a statistic: {} rounds to play'.format(num_games))
        self.num_games = num_games
        self.num_players = num_players
        self.num_agents = num_agents
        self.wins = np.zeros((num_games, num_players))

        if players is None:
            self.players = []
            # init average random players
            for _ in range(num_agents, num_players):
                self.players.append(AverageRandomPlayer())
            # init agents
            featurizer = Featurizer()
            estimator = DQNEstimator(input_shape=featurizer.get_state_size())
            trick_predictor = None
            if trick_prediction:
                trick_predictor = NNTrickPrediction()
            for _ in range(num_agents):
                self.players.append(RLAgent(estimator=estimator, featurizer=featurizer,
                                            trick_prediction=trick_predictor))
        else:
            self.players = players

    def play_games(self):
        for i in range(self.num_games):
            wiz = Wizard(num_players=self.num_players, players=self.players)
            scores = wiz.play()
            # evaluate scores
            index = np.argmax(scores)
            self.wins[i][index] = 1
            self.logger.info("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

    def plot_game_statistics(self):
        # plot:
        # x-Achse: gespielte Runden
        # y-Achse: Prozent gewonnene Spiele
        if self.num_games > 50:
            start = 50
        else:
            start = 0
        stat_num_games = np.arange(start, self.num_games, 1)
        fig, ax = plt.subplots()

        for i in range(len(self.players)):
            won_games = self.wins[:, i]

            won_stat = np.zeros(self.num_games - start, dtype=float)

            for game in range(start, self.num_games):
                if game < 1000:
                    won_stat[game - start] = np.sum(won_games[:game]) / (game + 1)
                else:
                    won_stat[game - start] = np.sum(won_games[game-1000:game]) / 1000

            ax.plot(stat_num_games, won_stat, color=self.plot_colors[i], label=self.get_playertype(self.players[i]))

        ax.set(xlabel='Number of rounds played', ylabel='Percentage of won games')
        ax.legend()
        # plt.show()
        name_plot = 'log/statistics/' + time.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        plt.savefig(name_plot)
        self.logger.info("Plot saved in {}".format(name_plot))

    def get_playertype(self, player):
        if isinstance(player, RLAgent):
            name = "RLAgent ({})".format(player.estimator.name_to_string())
        elif isinstance(player, AverageRandomPlayer):
            name = "AverageRandomPlayer"
        elif isinstance(player, RandomPlayer):
            name = "RandomPlayer"

        if player.trick_prediction is not None:
            name += " with Trick Prediction"

        return name

    def close(self):
        for player in self.players:
            player.close()


def init_logger():
    # create logger with 'wizard-rl'
    logger = logging.getLogger('wizard-rl')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs debug messages
    fh = logging.FileHandler('log/wizard-rl.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == "__main__":
    init_logger()

    tf.reset_default_graph()
    with tf.Session() as sess:
        featurizer = Featurizer()
        estimator = DQNEstimator(session=sess, input_shape=featurizer.get_state_size())
        pg_estimator = PolicyGradient(session=sess, input_shape=featurizer.get_state_size())
        trick_predictor = NNTrickPrediction(session=sess, name="Random")
        trick_predictor_random = NNTrickPrediction(session=sess)
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("log/graph", sess.graph)
        writer.close()

        players3 = [TrickPredictionRandomPlayer(trick_prediction=trick_predictor_random),
                    TrickPredictionRandomPlayer(trick_prediction=trick_predictor_random),
                    # RLAgent(estimator=estimator),
                    TrickPredictionRandomPlayer(trick_prediction=trick_predictor_random),
                    # AverageRandomPlayer(),
                    # RLAgent(estimator=pg_estimator, featurizer=featurizer),
                    # FunctionRandomPlayer()]
                    RLAgent(estimator=pg_estimator, trick_prediction=trick_predictor, featurizer=featurizer)]

        players2 = [AverageRandomPlayer(), AverageRandomPlayer(), AverageRandomPlayer(),
                    RLAgent(estimator=pg_estimator, featurizer=featurizer)]

        players = [AverageRandomPlayer(),
                   # RLAgent(estimator=estimator, trick_prediction=trick_predictor, featurizer=featurizer),
                   AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   RLAgent(estimator=estimator, featurizer=featurizer)]
        # players = [RLAgent(estimator=estimator, featurizer=featurizer) for _ in range(4)]

        stat = WizardStatistic(num_games=2000, num_agents=1, players=players)
        # stat = WizardStatistic(num_games=5000, num_agents=1, players=players2)
        # stat = WizardStatistic(num_games=5000, num_agents=1, players=players3)

        stat.play_games()
        stat.plot_game_statistics()

        del players
        del players2
        del players3
        del featurizer
        del estimator
        del pg_estimator
        del trick_predictor
        del trick_predictor_random
        del stat
