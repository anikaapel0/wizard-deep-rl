import tensorflow as tf
import numpy as np
import random

from Player import AverageRandomPlayer
from Wizard import Wizard


class TrickPrediction(object):
    n_hidden_1 = 40

    def __init__(self, input_shape=54, memory=10000, batch_size=1024, training_rounds=200):
        tf.reset_default_graph()
        self.input_shape = input_shape
        self.output_shape = 1
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
        self._session = None
        self._trained = False
        self._merged = None
        self._train_writer = None
        self.training_rounds = training_rounds
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="handcards")
            self._y = tf.placeholder("float", [None, self.output_shape], name="num_tricks")

        with tf.variable_scope("Trick_Prediction_Network"):
            hidden1 = tf.layers.dense(self._x, self.n_hidden_1, activation=tf.nn.relu, name="Hidden_1")
            self._prediction = tf.layers.dense(hidden1, self.output_shape, use_bias=False)

        with tf.variable_scope("Learning"):
            self._loss = tf.losses.mean_squared_error(self._y*10, self._prediction*10)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self._loss)

        tf.summary.scalar('loss_trick-prediction', self._loss)

        self._merged = tf.summary.merge_all()
        self._var_init = tf.global_variables_initializer()

        self._session = tf.Session()
        self._train_writer = tf.summary.FileWriter("log/trick-prediction/train-summary", self._session.graph)
        self._session.run(self._var_init)
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
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        summary, opt, loss = self._session.run([self._merged, self._optimizer, self._loss], feed_dict)
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

        # for i in range(self.training_rounds):
        #     wizard = Wizard(players=players, track_tricks=True)
        #     wizard.play()
        #
        #     x, y = wizard.get_history()
        #
        #     for j in range(x.shape[0]):
        #         self.update(x[j], None, y[j])
        #
        #     # temporärer Tracker
        #     if i % 100 == 0:
        #         print("Trick Prediction Initializer: Round {} finished".format(i))

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
