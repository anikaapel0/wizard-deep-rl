from RLAgents import RLAgent
from Player import RandomPlayer, AverageRandomPlayer
from Wizard import Wizard

import time
import numpy as np
import matplotlib.pyplot as plt


class WizardStatistic(object):
    plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']

    def __init__(self, num_games=20, num_players=4, num_agents=1, trick_prediction=False):
        self.num_games = num_games
        self.num_players = num_players
        self.num_agents = num_agents
        self.wins = np.zeros((num_games, num_players))
        self.players = []
        # init average random players
        for _ in range(num_agents, num_players):
            self.players.append(AverageRandomPlayer())
        # init agents
        for _ in range(num_agents):
            self.players.append(RLAgent(trick_prediction=trick_prediction))

    def play_games(self):
        for i in range(self.num_games):
            wiz = Wizard(num_players=self.num_players, players=self.players)
            scores = wiz.play()
            # evaluate scores
            index = np.argmax(scores)
            self.wins[i][index] = 1
            print("{0}: {1}".format(i, np.sum(self.wins, axis=0)))

    def plot_game_statistics(self, interval = 200):
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
    stat = WizardStatistic(num_games=5000, num_agents=1, trick_prediction=False)
    stat.play_games()
    stat.plot_game_statistics()
