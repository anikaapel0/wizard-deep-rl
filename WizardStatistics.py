from RLAgents import RLAgent
from Player import RandomPlayer, AverageRandomPlayer
from Wizard import Wizard
from Featurizers import Featurizer
from TrickPrediction import TrickPrediction
from Estimators import DQNEstimator

import numpy as np
import matplotlib.pyplot as plt

import gc
import psutil


class WizardStatistic(object):

    plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']

    def __init__(self, num_games=20, players=None, num_players=4, num_agents=1, trick_prediction=False):
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
                trick_predictor = TrickPrediction()
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
            print("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

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
        plt.show()

    def get_playertype(self, player):
        if isinstance(player, RLAgent):
            if player.trick_prediction is None:
                return "RLAgent"
            else:
                return "RL Agent with Trick Prediction"
        if isinstance(player, AverageRandomPlayer):
            return "AverageRandomPlayer"
        if isinstance(player, RandomPlayer):
            return "RandomPlayer"

    def close(self):
        for player in self.players:
            player.close()


if __name__ == "__main__":
    featurizer = Featurizer()
    estimator = DQNEstimator(input_shape=featurizer.get_state_size())
    trick_predictor = TrickPrediction()
    players = [AverageRandomPlayer(), AverageRandomPlayer(),
               RLAgent(estimator=estimator, featurizer=featurizer, trick_prediction=trick_predictor),
               RLAgent(estimator=estimator, featurizer=featurizer)]

    try:
        stat = WizardStatistic(num_games=5000, num_agents=2, players=players)
        stat.play_games()
        stat.plot_game_statistics()
    finally:
        if stat is not None:
            stat.close()

    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    gc.collect()
    print("After collecting:")
    print(psutil.virtual_memory())  # physical memory usage


