import time
from GamePlayer.Player import AverageRandomPlayer, PredictionRandomPlayer
from GamePlayer.TrickPrediction import TrickPrediction
from Environment.Wizard import Wizard


class TrickPredictionTrainer(object):

    def __init__(self, session, num_games=100000, interval=500):
        self.session = session
        self.num_games = num_games
        self.interval = interval
        self.path = "log/start_" + time.strftime("%Y-%m-%d_%H-%M-%S")
        self.name = "Trick-Prediction Training"
        self.num_player = 4
        self.tp_model = TrickPrediction(session, self.path)
        self.player = [PredictionRandomPlayer() for _ in range(self.num_player)]

    def train_tp(self):
        for i in range(self.num_games):
            game = Wizard(num_players=self.num_player, players=self.player, track_tricks=True)
            game.play()



