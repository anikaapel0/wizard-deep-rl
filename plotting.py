import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Player import AverageRandomPlayer, RandomPlayer
from RLAgents import RLAgent

plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']


def moving_average(interval, window_size):
    data = pd.DataFrame(interval)
    mvg_average = data.rolling(window=window_size).mean()
    return mvg_average.values


def plot_moving_average_wins(players, wins, scores, name, interval=25):
    # plot:
    # x-Achse: gespielte Runden
    # y-Achse: Prozent gewonnene Spiele
    num_games = wins.shape[0]
    stat_num_games = np.arange(0, num_games, 1)
    mvg_scores = moving_average(wins, interval)
    plt.figure(1)
    plt.subplot(211)

    for i in range(len(players)):
        won_games = wins[:, i]

        won_stat = np.zeros(num_games, dtype=float)


        for game in range(num_games):
            if 50 <= game < interval:
                won_stat[game] = np.sum(won_games[50:game]) / (game - 50 + 1)
            else:
                won_stat[game] = np.sum(won_games[game - interval:game]) / interval

        plt.plot(stat_num_games, won_stat, color=plot_colors[i], label=get_playertype(players[i]))
    # ax.set(xlabel='Number of rounds played', ylabel='Percentage of won games of last {} games'.format(interval))

    plt.subplot(212)
    moving_avg = moving_average(scores, interval)
    for i in range(len(players)):
        plt.plot(moving_avg[:, i], color=plot_colors[i], label=get_playertype(players[i]))

    # ax.set(xlabel='Number of rounds played', ylabel='Moving Average score')

    # plt.show()
    plt.savefig(name)


def plot_moving_average_scores(players, scores, name, window_size=25):
    fig, ax = plt.subplots()

    moving_avg = moving_average(scores, window_size)
    for i in range(len(players)):
        ax.plot(moving_avg[i], color=plot_colors[i], label=get_playertype(players[i]))

    ax.set(xlabel='Number of rounds played', ylabel='Moving Average score')
    plt.savefig(name)


def get_playertype(player):
    if isinstance(player, RLAgent):
        name = "RLAgent ({})".format(player.estimator.name)
        name = player.estimator.name()
    elif isinstance(player, AverageRandomPlayer):
        name = "AverageRandomPlayer"
    elif isinstance(player, RandomPlayer):
        name = "RandomPlayer"

    if player.trick_prediction is not None:
        name += " + Trick Pred"

    return name
