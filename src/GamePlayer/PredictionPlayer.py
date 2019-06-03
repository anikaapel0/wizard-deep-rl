from random import random, randrange, getrandbits
import logging

from GamePlayer import Featurizers

class PredictionPlayer(object):

    def __init__(self):
        self.logger = logging.getLogger('wizard-rl.PredictionPlayer')
        self.prediction = -1
        self.trick_prediction = None
        super(PredictionPlayer, self).__init__()

    def get_prediction(self, trump, predictions, players, game_num, restriction=None):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def check_restriction_random(self, prediction, restriction):
        if prediction == restriction:
            if getrandbits(1) == 1:
                prediction += 1
            else:
                prediction -= 1

        return prediction


class RandomPredictionPlayer(PredictionPlayer):
    def __init__(self):
        super(AveragePredictionPlayer, self).get_prediction()
    
    def get_prediction(self, trump, predictions, players, game_num, restriction=None):
        """Randomly return any number of wins between 0 and total number
         of games.
         """
        prediction = randrange(len(self.hand))

        self.prediction = self.check_restriction_random(prediction, restriction)
        return prediction


class AveragePredictionPlayer(PredictionPlayer):
    def __init__(self):
        super(AveragePredictionPlayer, self).__init__()

    def get_prediction(self, trump, predictions, players, game_num, restriction=None):
        prediction = len(self.hand) // len(predictions)
        self.prediction = self.check_restriction_random(prediction, restriction)

        return prediction


class NetworkPredictionPlayer(PredictionPlayer):
    def __init__(self):
        self.session = None
        self.featurizer = None
        self.trick_prediction = None
        super(NetworkPredictionPlayer, self).__init__()

    def init_prediction(self, session, trick_prediction, featurizer=None):
        self.session = session
        self.trick_prediction = trick_prediction
        if featurizer is None:
            self.featurizer = Featurizers.Featurizer()
        else:
            self.featurizer = featurizer

    def get_prediction(self, trump, predictions, players, game_num, restriction=None):
        s = self.featurizer.transform_handcards(self, trump)
        average = len(self.whole_hand) // len(players)
        prediction = self.trick_prediction.predict(s, average)

        # round prediction
        final_pred = min(max(0, int(round(prediction))), game_num)
        if restriction is not None and final_pred == restriction:
            if (prediction < final_pred and final_pred > 0) or final_pred == game_num:
                final_pred -= 1
            else:
                final_pred += 1
        # self.logger.info("Prediction: {}, Hand: {}, Trumpf: {}".format(final_pred, self.whole_hand, trump))

        return final_pred


