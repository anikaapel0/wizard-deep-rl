import tensorflow as tf
import numpy as np
import random

from Player import AverageRandomPlayer
from Card import Card
from Wizard import Wizard


class TrickPrediction(object):
    n_hidden_1 = 40

    def __init__(self, session, input_shape=59, memory=1000, batch_size=50, gamma=0.95):
        self.input_shape = input_shape
        self.output_shape = 1
        self.gamma = gamma
        self.memory = [([], 0, 0)] * memory
        self.batch_size = batch_size
        self.update_rate = max(1, batch_size // 8)
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
        self._train_writer = None
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="handcards")
            self._y = tf.placeholder("float", [None, self.output_shape], name="num_tricks")

        with tf.variable_scope("TP_Network"):
            hidden1 = tf.layers.dense(self._x, self.n_hidden_1, activation=tf.nn.relu, name="Hidden_1")
            self._prediction = tf.layers.dense(hidden1, self.output_shape)

        with tf.variable_scope("TP_Learning"):
            self._loss = tf.losses.mean_squared_error(self._y, self._prediction)
            self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)

        summary = tf.summary.scalar('loss_tp', self._loss)

        self._merged = tf.summary.merge([summary])

        self._train_writer = tf.summary.FileWriter("log/trick-prediction/train-summary", self._session.graph)
        self.print_graph()

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
        print("TRAINING TRICK PREDICTION no. {}".format(self.t_train))

        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        summary, opt, loss = self._session.run([self._merged, self._optimizer, self._loss], feed_dict)
        self._train_writer.add_summary(summary, self.t_train)

    def predict(self, s, average):
        if not self._trained:
            return average
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self._session.run(self._prediction, feed_dict)[0, 0]

    def close(self):
        if self._session is not None:
            self._session.close()

    def print_graph(self):
        graph = tf.get_default_graph()

        with tf.Session(graph=graph) as sess:
            writer = tf.summary.FileWriter("log/trick-prediction/graph", sess.graph)
            writer.close()
