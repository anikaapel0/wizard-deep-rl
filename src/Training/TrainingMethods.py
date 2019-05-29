from GamePlayer.RLAgents import RLAgent, PGAgent, DQNAgent, DoubleDQNAgent, DuelingAgent
from GamePlayer.Player import AverageRandomPlayer
from Environment.Wizard import Wizard
from Environment.Game import Game
from GamePlayer.TrickPrediction import TrickPrediction
from GamePlayer.Estimators.PolicyEstimators import PolicyGradient
from GamePlayer.Estimators.ValueEstimators import DuelingDQNEstimator, DoubleDQNEstimator, DQNEstimator
from GamePlayer.Featurizers import Featurizer

from Utils.plotting import plot_moving_average_wins

import logging
import time
import os
import numpy as np
import tensorflow as tf

DQN_PLAYER = "DQN"
DDQN_PLAYER = "DDQN"
DUELING_PLAYER = "DUELING"
PG_PLAYER = "PG"


class WizardTraining(object):
    def __init__(self, session, num_games=1000, interval=500, name="WizardTraining"):
        self.session = session
        self.num_games = num_games
        self.interval = interval
        self.path = "log/start_" + time.strftime("%Y-%m-%d_%H-%M-%S")
        self.name = name

    def _init_tracking(self):
        self.score_window = tf.placeholder("float", [None, self.num_players])
        self.wins_window = tf.placeholder("float", [None, self.num_players])
        self.curr_scores = tf.reduce_sum(self.score_window, axis=0) / self.evaluation_games
        self.curr_wins = tf.reduce_sum(self.wins_window, axis=0) / self.evaluation_games
        merging = []
        for i in range(self.num_players):
            name = self.players[i].name()
            merging.append(tf.summary.scalar('curr_score_{}_{}'.format(i, name), self.curr_scores[i]))
            merging.append(tf.summary.scalar('curr_wins_{}_{}'.format(i, name), self.curr_wins[i]))

        self._merged = tf.summary.merge(merging)
        self.score_writer = tf.summary.FileWriter(self.path, self.session.graph)

    def plot_game_statistics(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        filename = time.strftime("%Y-%m-%d_%H-%M-%S")

        plot_moving_average_wins(self.players, self.wins, self.scores, self.path + filename, interval=self.interval)

    def log_training_info(self):
        self.logger.info("Starting new {} with: ".format(self.name))
        log_players(self.logger, self.players)

    def disable_training(self):
        for player in self.players:
            if isinstance(player, RLAgent):
                player.disable_training()

    def enable_training(self):
        for player in self.players:
            if isinstance(player, RLAgent):
                player.enable_training()

    def train_agent(self):
        raise NotImplementedError("This method has to be implemented by your training class")


class SelfPlayTraining(WizardTraining):
    def __init__(self, session, players_type, tp=False, evaluation_players=None, num_games=1000, interval=500,
                 evaluation_games=1000):
        super().__init__(session=session, num_games=num_games, interval=interval, name="Self-Play Training")
        self.logger = logging.getLogger('wizard-rl.SelfPlayTraining')
        self.num_players = len(evaluation_players) if evaluation_players is not None else 4
        self.players = None
        self._init_players(players_type, tp)
        if evaluation_players is not None:
            self.evaluation_players = evaluation_players
        else:
            self.evaluation_players = [AverageRandomPlayer() for _ in range(self.num_players - 1)].append(
                self.players[0])
        self.evaluation_games = evaluation_games
        self.score_writer = None
        self.score_window = None
        self.wins_window = None
        self.t_eval = 0
        self._merged = None
        self._init_tracking()

    def _init_players(self, players_type, tp):
        trick_prediction = get_tp(self.session, tp)
        estimator, featurizer = get_estimator(self.session, players_type)
        self.players = []
        for _ in range(self.num_players):
            self.players.append(RLAgent(estimator=estimator, session=self.session, trick_prediction=trick_prediction,
                                        featurizer=featurizer))

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

    def train_agent(self, random_agents_interval):
        self.log_training_info()
        for i in range(self.num_games):
            if i % 100 == 0:
                self.logger.info("Playing training round {}".format(i))

            wiz = Wizard(num_players=self.num_players, players=self.players)
            wiz.play()

            if i % random_agents_interval == 0:
                self.play_evaluation_games()


class TrainingAgainstOtherPlayer(WizardTraining):

    def __init__(self, session, player_type, tp=False, opponents=None, num_games=1000, interval=500):
        super().__init__(session=session, num_games=num_games, interval=interval, name="Training against others")
        self.logger = logging.getLogger('wizard-rl.WizardStatistics.WizardStatistic')
        self.num_players = 4
        self.players = self.init_player(player_type, tp, opponents)
        self.wins = np.zeros((num_games, self.num_players))
        self.scores = np.zeros((num_games, self.num_players))
        self.t_train = 0
        self.score_writer = None
        self.score_window = None
        self.session = session
        self.wins_window = None
        self._merged = None
        self._init_tracking()

    def init_player(self, player_type, tp, opponents):
        if opponents is None:
            player = [AverageRandomPlayer() for _ in range(self.num_players)]
        else:
            player = opponents

        player.append(get_player(player_type, tp))

        return player

    def update_scores(self, scores, i):
        self.scores[i] = scores
        self.wins[i][np.argmax(scores)] = 1

        if i > self.interval:
            curr_scores = self.scores[self.t_train - self.interval: self.t_train]
            curr_wins = self.wins[self.t_train - self.interval: self.t_train]
            summary, mean_wins, mean_scores = self.session.run([self._merged, self.curr_wins, self.curr_scores],
                                                               feed_dict={self.score_window: curr_scores,
                                                                          self.wins_window: curr_wins})

        self.score_writer.add_summary(summary, self.t_train)

    def get_winner(self):
        last_wins = np.sum(self.wins[-1000:], axis=0)

        return self.players[np.argmax(last_wins)]

    def train_agent(self):
        self.log_training_info()
        for i in range(self.num_games):
            if i % 100 == 0:
                self.logger.info("Playing training round {}".format(i))

            wiz = Wizard(num_players=self.num_players, players=self.players)
            scores = wiz.play()

            self.update_scores(scores, i)


def get_tp(session, tp):
    if tp:
        return TrickPrediction(session=session)
    else:
        return None


def get_estimator(session, player_type):
    featurizer = Featurizer()
    input_shape = featurizer.get_state_size()
    if player_type == DQN_PLAYER:
        estimator = DQNEstimator(session=session, input_shape=input_shape)
    elif player_type == DDQN_PLAYER:
        estimator = DoubleDQNEstimator(session=session, input_shape=input_shape)
    elif player_type == PG_PLAYER:
        estimator = PolicyGradient(session=session, input_shape=input_shape)
    elif player_type == DUELING_PLAYER:
        estimator = DuelingDQNEstimator(session=session, input_shape=input_shape)

    return estimator, featurizer


def get_player(session, player_type, tp=None):
    trick_prediction = get_tp(session, tp)
    if player_type == DQN_PLAYER:
        return DQNAgent(session=session, trick_prediction=trick_prediction)
    elif player_type == DDQN_PLAYER:
        return DoubleDQNAgent(session=session, trick_prediction=trick_prediction)
    elif player_type == PG_PLAYER:
        return PGAgent(session=session, trick_prediction=trick_prediction)
    elif player_type == DUELING_PLAYER:
        return DuelingAgent(session=session, trick_prediction=trick_prediction)


def log_players(logger, players):
    for player in players:
        logger.info("\t{}".format(player.name()))
