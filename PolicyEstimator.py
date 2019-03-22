from Card import Card
from Estimators import Estimator

import tensorflow as tf
import numpy as np


class PolicyGradient(Estimator):
    n_hidden_1 = 128

    def __init__(self, session, input_shape, output_shape=Card.DIFFERENT_CARDS, gamma=0.99, update=1000):
        self.memory = []
        self.memory_temp = []
        self.t = 0
        self.update_rate = update
        self.session = session
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma  # discount factor
        self._x = None
        self._y = None
        self._logits = None
        self._loss = None
        self._optimizer = None
        self._merged = None
        self._sum_writer = None
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("PG_Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="pg_state")
            self._y = tf.placeholder("float", [None, self.output_shape], name="pg_state")

        with tf.variable_scope("PG_Network"):
            hidden1 = tf.layers.dense(self._x, self.n_hidden_1, name="PG_Hidden_1",
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            self._logits = tf.layers.dense(hidden1, self.output_shape, name="PG_Output",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.sigmoid(self._logits, name="sigmoid")

        with tf.variable_scope("PG_Learning"):
            self._loss = tf.losses.sigmoid_cross_entropy(self._y, self._logits)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self._loss)

        tf.summary.scalar('loss_policy-gradient', self._loss)
        tf.summary.histogram("hidden_out", hidden1)
        tf.summary.histogram("prob_out", out)
        self._merged = tf.summary.merge([self._loss, hidden1, out])
        self._sum_writer = tf.summary.FileWriter("log/pg/train-summary", self._session.graph)

    def update(self, s, a, r, s_prime):
        self.memory_temp.append([s, a, r, s_prime])

        # trick ended, update memory
        if s_prime is None:
            self.update_after_round(r)

    def update_after_round(self, reward):
        self.t += 1

        i = 0
        reward_discounted = 0
        for s, a, r, s_prime in reversed(self.memory_temp):
            reward_discounted = reward_discounted * self.gamma + r  # ist r nicht immer 0 außer im letzten?
            self.memory.append([s, a, i, reward_discounted])
            i += 1

        self.memory_temp = []

        if self.t % self.update_rate == 0:
            self.update_model()

    def update_model(self):
        num_sets = len(self.memory)

        print("PG-Model updated with {}".format(num_sets))
        x = np.zeros((num_sets, self.input_shape))
        y = np.zeros((num_sets, self.output_shape))

        i = 0
        for s, a, t, reward in self.memory:
            x[i] = s
            y[i][int(a)] = reward
            i += 1

        # reset memory
        self.memory = []
        # update model
        self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        opt, loss = self.session.run([self._optimizer, self._loss], feed_dict=feed_dict)

    def predict(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        logits = self.session.run(self._logits, feed_dict)
        return logits

    def save(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def load(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")
