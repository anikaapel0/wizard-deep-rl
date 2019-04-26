import numpy as np
import tensorflow as tf
import logging

from Estimators.Estimators import Estimator
from GameUtilities.Card import Card


class PolicyGradient(Estimator):
    n_hidden_1 = 500
    n_hidden_2 = 250

    def __init__(self, session, input_shape, output_shape=Card.DIFFERENT_CARDS, gamma=0.99, update=1000, batch_size=500):
        self.logger = logging.getLogger('PolicyGradient')
        self.memory = []
        self.memory_temp = []
        self.t = 0
        self.t_train = 0
        self.update_rate = update
        self.session = session
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma  # discount factor
        self.batch_size = batch_size
        self._x = None
        self._actions = None
        self._rewards = None
        self._logits = None
        self._probs = None
        self._loss = None
        self._optimizer = None
        self._merged = None
        self._sum_writer = None
        self._init_model()

    def _init_model(self):
        with tf.variable_scope("PG_Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="pg_state")
            self._actions = tf.placeholder("float", [None, self.output_shape], name="pg_state")
            self._rewards = tf.placeholder("float", [None, self.output_shape], name="rewards")

        with tf.variable_scope("PG_Network"):
            hidden1 = tf.layers.dense(self._x, self.n_hidden_1, name="PG_Hidden_1",
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, self.n_hidden_2, name="PG_Hidden_2",
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            self._logits = tf.layers.dense(hidden2, self.output_shape, name="PG_Output",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.sigmoid(self._logits, name="sigmoid")
            self._probs = tf.nn.softmax(self._logits)

        with tf.variable_scope("PG_Learning"):
            cross_entropy = tf.losses.sigmoid_cross_entropy(self._actions, self._logits)
            self._loss = tf.reduce_sum(tf.multiply(self._rewards, cross_entropy))
            self._optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)

        sum_loss = tf.summary.scalar('loss_policy-gradient', self._loss)
        sum_hidden_w = tf.summary.histogram("hidden_out", hidden1)
        sum_out_w = tf.summary.histogram("prob_out", out)
        self._merged = tf.summary.merge([sum_loss, sum_hidden_w, sum_out_w])
        self._sum_writer = tf.summary.FileWriter("log/pg/train-summary", self.session.graph)

    def update(self, s, a, r, s_prime):
        self.memory_temp.append([s, a, r, s_prime])

        # trick ended, update memory
        if s_prime is None:
            self.update_after_round()

    def update_after_round(self):
        self.t += 1

        reward_discounted = 0
        for s, a, r, s_prime in reversed(self.memory_temp):
            reward_discounted = reward_discounted * self.gamma + r // 10    # ist r nicht immer 0 außer im letzten?
            self.memory.append([s, a, reward_discounted])

        # reset temporary memory
        del self.memory_temp
        self.memory_temp = []

        if self.t % self.update_rate == 0:
            self.update_model()

    def update_model(self):
        num_sets = len(self.memory)
        # self.logger.info("PG-Model updated with {} Rounds played".format(num_sets))
        self.logger.info("PG-Model updated with {} Rounds played".format(num_sets))

        x = np.zeros((num_sets, self.input_shape))
        y = np.zeros((num_sets, self.output_shape))
        rewards = np.zeros((num_sets, self.output_shape))

        i = 0
        for s, a, reward in self.memory:
            x[i] = s
            y[i][a] = 1
            rewards[i][a] = reward
            i += 1

        pos = 0
        while pos < num_sets:
            end = num_sets if pos + self.batch_size >= num_sets else pos + self.batch_size
            # update model
            self.train_model(x[pos:end], y[pos:end], rewards[pos:end])
            pos = end

        # reset memory
        del self.memory
        self.memory = []

    def train_model(self, batch_x, batch_y, batch_r):
        self.t_train += 1
        feed_dict = {
            self._x: batch_x,
            self._actions: batch_y,
            self._rewards: batch_r
        }
        summary, opt, loss = self.session.run([self._merged, self._optimizer, self._loss], feed_dict=feed_dict)

        self._sum_writer.add_summary(summary, self.t_train)

    def predict(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        probs, logits = self.session.run([self._probs, self._logits], feed_dict)
        return probs

    def save(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def load(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def name(self):
        return "PolicyGradient"
