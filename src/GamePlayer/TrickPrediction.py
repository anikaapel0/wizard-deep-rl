import tensorflow as tf
import numpy as np
import random
import logging

from GamePlayer.Player import AverageRandomPlayer
from Environment.Wizard import Wizard
from Environment.Wizard import MAX_ROUNDS
from Environment.Card import Card


class TrickPrediction(object):
    n_hidden_1 = 40

    def __init__(self, session, path, input_shape=59, memory=10000, batch_size=1024, training_rounds=200,
                 learning_rate=0.005, save_update=500):
        self.logger = logging.getLogger('wizard-rl.TrickPrediction')
        self.path = path
        self.input_shape = input_shape
        self.output_shape = 1
        self.learning_rate = learning_rate
        self.memory = [([], 0, 0)] * memory
        self.batch_size = batch_size
        self.update_rate = max(1, batch_size // 8)
        self.save_update = save_update
        self.saver = None
        self.t = 0
        self.t_train = 0
        self._prediction = None
        self._optimizer = None
        self._loss = None
        self._x = None
        self._y = None
        self._var_init = None
        self._session = session
        self._trained = False
        self._merged = None
        self._histos = [None] * MAX_ROUNDS
        self._sum_histograms = [None] * MAX_ROUNDS
        self._train_writer = None
        self.training_rounds = training_rounds
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("TP_Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="handcards")
            self._y = tf.placeholder("float", [None, self.output_shape], name="num_tricks")

        with tf.variable_scope("TP_Network"):
            hidden1 = tf.layers.dense(self._x, self.n_hidden_1, activation=tf.nn.relu, name="Hidden_1")
            self._prediction = tf.layers.dense(hidden1, self.output_shape, use_bias=False)

        with tf.variable_scope("TP_Learning"):
            self._loss = tf.losses.mean_squared_error(self._y*10, self._prediction*10)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

        # tracking of trick prediction loss
        summary = tf.summary.scalar('loss_tp', self._loss)
        self._merged = tf.summary.merge([summary])

        # histogramm of trick prediction results per game round
        for i in range(MAX_ROUNDS):
            self._histos[i] = tf.summary.histogram("histo_tp_{}_cards".format(i),
                                                   tf.math.round(tf.reduce_sum(self._prediction)))
            self._sum_histograms[i] = tf.summary.merge([self._histos[i]])

        self._train_writer = tf.summary.FileWriter(self.path, self._session.graph)

        self.saver = tf.train.Saver()

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
            self.update_network()

    def create_minibatch(self):
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

        return x, y

    def update_network(self):
        self._trained = True
        for _ in range(50):
            x, y = self.create_minibatch()

            self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        self.t_train += 1
        self.logger.info("TRAINING TRICK PREDICTION no. {}".format(self.t_train))
        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        summary, opt, loss = self._session.run([self._merged, self._optimizer, self._loss], feed_dict)
        self.logger.info("Epoch {} - Loss: {}".format(self.t_train, loss))
        self._train_writer.add_summary(summary, self.t_train)

        if self.t_train % self.save_update == 0:
            self.save()

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
                self.logger.info("Trick Prediction Initializer: Round {} finished".format(i))

        return x, y

    def init_training(self, num_players=4, epochs=100):
        self.logger.info("Initial training for trick prediction")

        players = [AverageRandomPlayer() for _ in range(num_players)]

        x, y = self.collect_training_data(players)

        batch_size = 1024

        for e in range(epochs):
            batch_idx = np.random.choice(np.arange(len(x)), batch_size, replace=False)
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]

            self.train_model(batch_x, batch_y[:, np.newaxis])

        self.logger.info("Initial Training finished")
        self._trained = True

    def predict(self, s, average):
        game = np.sum(s[:Card.DIFFERENT_CARDS])
        if not self._trained:
            # self.init_training()
            return average

        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        summ, histo, prediction = self._session.run([self._sum_histograms[game], self._histos[game], self._prediction], feed_dict)
        self._train_writer.add_summary(summ)

        return prediction[0, 0]

    def save(self):
        save_path = self.saver.save(self.session, self.path + "/models/model_tp.ckpt")
        self.logger.info("{}: Model saved in {}".format(self.name(), save_path))


