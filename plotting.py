import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Player import AverageRandomPlayer, RandomPlayer
from RLAgents import RLAgent

plot_colors = ['b', 'k', 'r', 'c', 'm', 'y']


def moving_average(interval, window_size, name = None):
    data = pd.DataFrame(interval)
    if name is not None:
        data.to_csv(name, sep=";")
    mvg_average = data.rolling(window=window_size).mean()
    return mvg_average.values


def plot_moving_average_wins(players, wins, scores, name, interval=25):
    # plot:
    # x-Achse: gespielte Runden
    # y-Achse: Prozent gewonnene Spiele
    num_games = wins.shape[0]
    stat_num_games = np.arange(0, num_games, 1)
    mvg_scores = moving_average(wins, interval, name + "_scores.csv")
    moving_avg = moving_average(scores, interval, name + "_wins.csv")

    f, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

    for i in range(len(players)):
        ax1.plot(mvg_scores[:, i], color=plot_colors[i], label=get_playertype(players[i]))
    ax1.legend()
    ax1.set(xlabel='Number of rounds played', ylabel='Moving average of wins (window = {})'.format(interval))

    for i in range(len(players)):
        ax2.plot(moving_avg[:, i], color=plot_colors[i], label=get_playertype(players[i]))
    ax2.set(xlabel='Number of rounds played', ylabel='Moving average of scores (window = {})'.format(interval))
    ax2.legend()
    # plt.show()
    plt.savefig(name + "_plot.png")


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
