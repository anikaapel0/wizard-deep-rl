import tensorflow as tf
import numpy as np
import random


class TrickPrediction(object):
    n_hidden_1 = 256

    def __init__(self, input_shape = 54, memory=100, batch_size=50, gamma=0.95):
        tf.reset_default_graph()
        self.input_shape = input_shape
        self.output_shape = 21  # up to 20 tricks in a round is possible + zero tricks
        self.gamma = gamma
        self.memory = [([], 0, 0, [])] * memory
        self.batch_size = batch_size
        self.update_rate = max(1, batch_size // 8)
        self.t = 0
        self._prediction = None
        self._optimizer = None
        self._loss = None
        self._x = None
        self._y = None
        self._var_init = None
        self._session = None
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="handcards")
            self._y = tf.placeholder("float", [None, self.output_shape], name="no_tricks")

        with tf.variable_scope("Trick_Prediction_Network"):
            hidden1 = tf.layers.dense(self._x, self.n_hidden_1, activation=tf.nn.relu, name="Hidden_1")
            self._prediction = tf.layers.dense(hidden1, self.output_shape)

        with tf.variable_scope("Learning"):
            self._loss = tf.losses.mean_squared_error(self._y, self._prediction)
            self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)

        self._var_init = tf.global_variables_initializer()

        self._session = tf.Session()
        self._session.run(self._var_init)

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
        if self.t >= len(self.memory) and self.t % self.update_rate == 0:
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
                y[i, num_tricks] = 1
                i += 1

            self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        self._session.run([self._optimizer, self._loss], feed_dict)

    def predict(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self._session.run(self._prediction, feed_dict)

    def close(self):
        if self._session is not None:
            self._session.close()
