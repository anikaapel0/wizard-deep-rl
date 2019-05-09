from GameUtilities.Game import Game
from Player import AverageRandomPlayer
from GameUtilities.Card import Card
from Featurizers import Featurizer

from random import seed, getstate, choice
import numpy as np
import logging


class Wizard(object):
    """

    """
    NUM_CARDS = 60

    def __init__(self, num_players=4, players=None, track_tricks=False):
        self.logger = logging.getLogger('wizard-rl.Wizard')

        self.players = []
        if players is None:
            assert num_players >= 3, "Not enough players!" \
                                     "Give an array of players or a" \
                                     "number of players between [3-6]"

            for player in range(num_players):
                # Initialize all players
                # self.logger.info("Creating players.")
                self.players.append(AverageRandomPlayer())
        else:
            self.players = players
        self.num_players = len(self.players)
        self.games_to_play = Wizard.NUM_CARDS // self.num_players
        self.scores = [0] * self.num_players
        self.random_start = choice(np.arange(0, self.num_players, 1))
        self.track_tricks = track_tricks
        self.featurizer = Featurizer()
        self.history = [[] for _ in range(2)]
        self.history[0] = np.zeros((self.games_to_play * num_players, Card.DIFFERENT_CARDS))
        self.history[1] = np.zeros(self.games_to_play * num_players)

    def play(self):
        """
        Starts a game with the generated players.

        Returns:
            list: The scores for each player.

        """
        # self.logger.info("Playing a Wizard game!")
        for game_num in range(1, self.games_to_play+1):
            game = Game(game_num, self.players, self.random_start)
            score = game.play()
            for i in range(self.num_players):
                self.scores[i] += score[i]

            if self.track_tricks:
                for i in range(len(self.players)):
                    player = self.players[i]
                    curr_idx = self.num_players * (game_num - 1) + i
                    self.history[1][curr_idx] = player.wins
                    self.history[0][curr_idx] = self.featurizer.transform_handcards(player, game.trump_card)
            # self.logger.info("Scores: {}".format(self.scores))
        # self.logger.info("Final scores: {}".format(self.scores))
        for player in self.players:
            player.reset_score()
        return self.scores

    def get_history(self):
        return self.history[0], self.history[1]


if __name__ == "__main__":
    print("Playing a random game of 4 players.")
    seed(2)
    wiz = Wizard(4)
    print(getstate())
    print(wiz.play())
