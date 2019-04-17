from RLAgents import RLAgent
from Player import RandomPlayer, AverageRandomPlayer
from Wizard import Wizard
from ValueEstimators import DQNEstimator, DoubleDQNEstimator
from Featurizers import Featurizer
from TrickPrediction import TrickPrediction

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class WizardStatistic(object):
    plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']

    def __init__(self, players, num_games=20):
        self.num_games = num_games
        self.num_players = len(players)
        self.wins = np.zeros((num_games, len(players)))
        self.players = players

    def play_games(self):
        for i in range(self.num_games):
            wiz = Wizard(num_players=self.num_players, players=self.players)
            scores = wiz.play()
            # evaluate scores
            index = np.argmax(scores)
            self.wins[i][index] = 1
            print("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

    def plot_game_statistics(self, interval=200):
        # plot:
        # x-Achse: gespielte Runden
        # y-Achse: Prozent gewonnene Spiele
        stat_num_games = np.arange(0, self.num_games, 1)
        fig, ax = plt.subplots()

        for i in range(len(self.players)):
            won_games = self.wins[:, i]

            won_stat = np.zeros(self.num_games, dtype=float)

            for game in range(self.num_games):
                if 50 <= game < interval:
                    won_stat[game] = np.sum(won_games[50:game]) / (game - 50 + 1)
                else:
                    won_stat[game] = np.sum(won_games[game - interval:game]) / interval

            ax.plot(stat_num_games, won_stat, color=self.plot_colors[i], label=self.get_playertype(self.players[i]))

        ax.set(xlabel='Number of rounds played', ylabel='Percentage of won games of last {} games'.format(interval))
        ax.legend()
        name_plot = 'log/statistics/' + time.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        plt.savefig(name_plot)

    def get_playertype(self, player):
        if isinstance(player, RLAgent):
            return "RLAgent"
        if isinstance(player, AverageRandomPlayer):
            return "AverageRandomPlayer"
        if isinstance(player, RandomPlayer):
            return "RandomPlayer"


if __name__ == "__main__":
    tf.reset_default_graph()

    with tf.Session() as sess:
        featurizer = Featurizer()
        estimator = DQNEstimator(sess, input_shape=featurizer.get_state_size())
        double_estimator = DoubleDQNEstimator(sess, input_shape=featurizer.get_state_size())
        tp = TrickPrediction(sess)

        sess.run(tf.global_variables_initializer())

        players = [AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   AverageRandomPlayer(),
                   RLAgent(featurizer=featurizer, estimator=double_estimator)]

        stat = WizardStatistic(players, num_games=5000)
        stat.play_games()
        stat.plot_game_statistics()
