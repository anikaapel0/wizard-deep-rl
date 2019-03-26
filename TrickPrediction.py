import tensorflow as tf
import numpy as np
import random
import logging

from Player import AverageRandomPlayer
from Wizard import Wizard


class TrickPrediction(object):
    n_hidden_1 = 40

    def __init__(self, session, input_shape=59, memory=5000, batch_size=512, training_rounds=200):
        self.logger = logging.getLogger('wizard-rl.TrickPrediction.TrickPrediction')
        self.input_shape = input_shape
        self.output_shape = 1
        self.memory = [([], 0, 0)] * memory
        self.batch_size = batch_size
        self.update_rate = max(1, batch_size // 8)
        self.t = 0
        self.t_train = 0
        self._trick_prediction = None
        self._trick_optimizer = None
        self._trick_loss = None
        self._features = None
        self._tricks = None
        self._var_init = None
        self._session = session
        self._trained = False
        self._merged2 = None
        self._train_writer = None
        self.training_rounds = training_rounds
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("Input_Data_TrickPrediction"):
            self._features = tf.placeholder("float", [None, self.input_shape], name="handcards")
            self._tricks = tf.placeholder("float", [None, self.output_shape], name="num_tricks")

        with tf.variable_scope("Trick_Prediction_Network"):
            hidden1_trick = tf.layers.dense(self._features, self.n_hidden_1, activation=tf.nn.relu, name="Trick_Hidden_1")
            self._trick_prediction = tf.layers.dense(hidden1_trick, self.output_shape, use_bias=False)

        with tf.variable_scope("Learning_TrickPrediction"):
            self._trick_loss = tf.losses.mean_squared_error(self._tricks, self._trick_prediction)
            self._trick_optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self._trick_loss)

        sum_trick_loss = tf.summary.scalar('loss_trick-prediction', self._trick_loss)

        self._merged2 = tf.summary.merge([sum_trick_loss])
        self._var_init = tf.global_variables_initializer()

        self._train_writer = tf.summary.FileWriter("log/trick-prediction/train-summary", self._session.graph)

    def update(self, cards, num_forecast, num_tricks):
        """
        Fills one entry in the memory and updates the estimator.
        Args:
            cards: np.array of handcards
            num_forecast: number of tricks player declared
            num_tricks: number of tricks player got
        """
        # Circular buffer for memory.
        self.memory[self.t % len(self.memory)] = (cards, num_forecast, num_tricks)
        self.t += 1

        if self.t == len(self.memory) * 2:
            # Prevent overflow, this might cause skidding in the update rate
            self.t = len(self.memory)

        # If memory is full, we can start training
        if self.t % len(self.memory) == 0:
            print("-----------TRICK-PREDICTION TRAINED-----------")
            self._trained = True
            for _ in range(50):
                # Randomly sample from experience
                minibatch = random.sample(self.memory, self.batch_size)
                # Initialize x and y for the neural network
                x = np.zeros((self.batch_size, self.input_shape))
                y = np.zeros((self.batch_size, self.output_shape))

                # Iterate over the minibatch to fill x and y
                i = 0
                for card, num_forecast, num_tricks in minibatch:
                    # x are simply the hand cards of player
                    x[i] = card
                    y[i] = num_tricks
                    i += 1

                self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        self.t_train += 1
        feed_dict = {
            self._features: batch_x,
            self._tricks: batch_y
        }
        # train network
        summary, opt, loss = self._session.run([self._merged2, self._trick_optimizer, self._trick_loss], feed_dict)
        print("Epoch {} - Loss: {}".format(self.t_train, loss))
        self._train_writer.add_summary(summary, self.t_train)

    def collect_training_data(self, players):
        x = None
        y = None

        for i in range(self.training_rounds):
            wizard = Wizard(players=players, track_tricks=True)
            wizard.play()

            temp_x, temp_y = wizard.get_history()
            if x is None:
                x = temp_x
                y = temp_y
            else:
                x = np.concatenate((x, temp_x), axis=0)
                y = np.concatenate((y, temp_y), axis=0)

            # temporärer Tracker
            if i % 100 == 0:
                print("Trick Prediction Initializer: Round {} finished".format(i))

        return x, y

    def init_training(self, num_players=4, epochs=100):
        print("Initial training for trick prediction")

        players = [AverageRandomPlayer() for _ in range(num_players)]

        x, y = self.collect_training_data(players)

        batch_size = 1024

        for e in range(epochs):
            batch_idx = np.random.choice(np.arange(len(x)), batch_size, replace=False)
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]

            self.train_model(batch_x, batch_y[:, np.newaxis])

        print("Initial Training finished")
        self._trained = True

    def predict(self, s, average):
        if not self._trained:
            # self.init_training()
            return average

        feed_dict = {self._features: np.array(s)[np.newaxis, :]}
        return self._session.run(self._trick_prediction, feed_dict)[0, 0]

    def close(self):
        if self._session is not None:
            self._session.close()
