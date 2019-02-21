from RLAgents import RLAgent
from Player import RandomPlayer, AverageRandomPlayer
from Wizard import Wizard

import numpy as np
import matplotlib.pyplot as plt


class WizardStatistic(object):

    plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']

    def __init__(self, num_games=20, num_players=4, num_agents=1):
        self.num_games = num_games
        self.num_players = num_players
        self.num_agents = num_agents
        self.wins = np.zeros((num_games, num_players))
        self.players = []
        # init agents
        for _ in range(num_agents):
            self.players.append(RLAgent())

        # init average random players
        for _ in range(num_agents, num_players):
            self.players.append(AverageRandomPlayer())

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
        # Average Random Player: grau
        # RL Agent: green
        stat_num_games = np.arange(1, self.num_games + 1, 1)
        fig, ax = plt.subplots()

        for i in range(len(self.players)):
            won_games = self.wins[:, i]

            won_stat = np.zeros(self.num_games, dtype=float)

            for game in range(self.num_games):
                won_stat[game] = np.sum(won_games[:game]) / (game + 1)

            ax.plot(stat_num_games, won_stat, color=self.plot_colors[i], label=self.get_playertype(self.players[i]))

        #plt.xticks(stat_num_games)
        ax.set(xlabel='Number of rounds played', ylabel='Percentage of won games')
        ax.legend()
        plt.show()

    def get_playertype(self, player):
        if isinstance(player, RLAgent):
            return "RLAgent"
        if isinstance(player, AverageRandomPlayer):
            return "AverageRandomPlayer"
        if isinstance(player, RandomPlayer):
            return "RandomPlayer"


if __name__ == "__main__":
    stat = WizardStatistic(num_games=1000)
    stat.play_games()
    stat.plot_game_statistics()

