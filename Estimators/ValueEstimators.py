import tensorflow as tf
import numpy as np
import random
import logging

from Estimators.Estimators import Estimator
from GameUtilities.Card import Card
from GameUtilities.Card import cards_to_bool_array


class ValueEstimator(Estimator):

    def __init__(self, session, path, input_shape, output_shape=Card.DIFFERENT_CARDS, memory=100000, batch_size=1024,
                 target=False, target_update=1000):
        super(ValueEstimator, self).__init__()
        self.session = session
        self.memory = [([], 0, 0, [])] * memory
        self.t = 0
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.target_update = target_update
        self.use_target = target
        self.batch_size = batch_size
        self.update_rate = max(1, batch_size // 8)
        self.saver = None
        self.path = path

    def update(self, s, a, r, s_prime):
        """
        Fills one entry in the memory and updates the estimator.
        Args:
            s: np.array state where the action was taken
            a: int action taken at this timestep
            r: int reward after taking action a
            s_prime: np.array state after taking action a

        """

        # Circular buffer for memory.
        self.memory[self.t % len(self.memory)] = (s, a, r, s_prime)
        self.t += 1
        if self.t == len(self.memory) * 2:
            # Prevent overflow, this might cause skidding in the update rate
            self.t = len(self.memory)

        if self.use_target and self.t >= len(self.memory) and self.t % self.target_update == 0:
            self.update_target()
        # If memory is full, we can start training
        if self.t >= len(self.memory) and self.t % self.update_rate == 0:
            self.update_from_experience()

    def update_from_experience(self):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class.")

    def update_target(self):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class.")

    def save(self):
        save_path = self.saver.save(self.session, self.path + "/models/model.ckpt")
        self.logger.info("{}: Model saved in {}".format(self.name(), save_path))


class DQNEstimator(ValueEstimator):
    n_hidden_1 = 1024
    n_hidden_2 = 512
    n_hidden_3 = 256

    def __init__(self, session, input_shape, limit_update=False, output_shape=Card.DIFFERENT_CARDS, memory=100000,
                 batch_size=1024, gamma=0.95, target_update=1000, save_update=5000, path="log/dqn"):
        super(DQNEstimator, self).__init__(session, path, input_shape, output_shape, memory, batch_size, True, target_update)
        self.logger = logging.getLogger('wizard-rl.DQNEstimator')
        self.gamma = gamma
        self.learning_rate = 0.0005
        self.limit_update = limit_update
        self.save_update = save_update
        self._prediction = None
        self._optimizer = None
        self._loss = None
        self._x = None
        self._y = None
        self._target = None
        self.counter_train = 0
        self._sum_writer = None
        self._merged = None
        self._init_model()

    def dqn_network(self, input_size, scope_name, act=tf.nn.relu):

        with tf.variable_scope(scope_name):
            hidden1 = tf.layers.dense(input_size, self.n_hidden_1, activation=act, name=scope_name + "_1",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, self.n_hidden_2, activation=act, name=scope_name + "_2",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden3 = tf.layers.dense(hidden2, self.n_hidden_3, activation=act, name=scope_name + "_3",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

            prediction = tf.layers.dense(hidden3, self.output_shape)

        return prediction

    def _init_model(self):
        with tf.variable_scope("Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="state")
            self._y = tf.placeholder("float", [None, self.output_shape], name="output")

        self._prediction = self.dqn_network(input_size=self._x, scope_name="Q_Primary")
        self._target = self.dqn_network(input_size=self._x, scope_name="Q_Target")

        with tf.variable_scope("Learning"):
            self._loss = tf.losses.mean_squared_error(self._y, self._prediction)
            # self._loss = tf.losses.huber_loss(self._y, self._prediction)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

        summary = tf.summary.scalar('loss_dqn', self._loss)

        self._merged = tf.summary.merge([summary])
        self._sum_writer = tf.summary.FileWriter(self.path, self.session.graph)

        self.saver = tf.train.Saver()
        # self.saver = tf.train.Saver({'Q_Primary': self._prediction, 'Q_Target': self._target})

    def update_from_experience(self):
        # Randomly sample from experience
        minibatch = random.sample(self.memory, self.batch_size)
        # Initialize x and y for the neural network
        x = np.zeros((self.batch_size, self.input_shape))
        y = np.zeros((self.batch_size, self.output_shape))

        # Iterate over the minibatch to fill x and y
        i = 0
        for ss, aa, rr, ss_prime in minibatch:
            # x is simply the state
            x[i] = ss
            # y are the q values for each action.
            y[i] = self.predict(ss)

            # We update the action taken ONLY.
            if ss_prime is not None:
                # ss_prime is not None, so this is not a terminal state.
                if self.limit_update:
                    playable_bool = cards_to_bool_array(ss_prime[:Card.DIFFERENT_CARDS])
                    q_sa = self.predict_target(ss_prime)
                    q_sa = q_sa[:, playable_bool]
                else:
                    q_sa = self.predict_target(ss_prime)
                y[i, aa] = rr / 10 + self.gamma * np.max(q_sa)
            else:
                # ss_prime is None so this is a terminal state.
                y[i, aa] = rr / 10
            i += 1

        self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        self.counter_train += 1

        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        summary, _, loss = self.session.run([self._merged, self._optimizer, self._loss], feed_dict)

        self._sum_writer.add_summary(summary, self.counter_train)

        if self.counter_train % self.save_update == 0:
            self.save()

    def predict(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self.session.run(self._prediction, feed_dict)

    def predict_target(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self.session.run(self._target, feed_dict)

    def update_target(self):
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Primary")
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Target")

        assert len(q_vars) == len(q_target_vars)

        # hard update
        self.session.run([v_t.assign(v) for v_t, v in zip(q_target_vars, q_vars)])

    def load(self, name="model-dqn"):
        self.saver.restore(self.session, self.path + "/models/model.ckpt")

    def name(self):
        return "DQN"


class DoubleDQNEstimator(ValueEstimator):
    n_hidden_1 = 256
    n_hidden_2 = 512
    n_hidden_3 = 1024

    def __init__(self, session, input_shape, output_shape=Card.DIFFERENT_CARDS, memory=100000, batch_size=1024, gamma=0.95,
                 target_update=1000, save_update=5000, path="log"):
        super(DoubleDQNEstimator, self).__init__(session, path, input_shape, output_shape, memory, batch_size, True, target_update)
        self.logger = logging.getLogger('wizard-rl.DoubleDQNEstimator')
        self.gamma = gamma
        self.update_rate = max(1, batch_size // 8)
        self.save_update = save_update
        self._prediction = None
        self._optimizer = None
        self._loss = None
        self._x = None
        self._y = None
        self._target = None
        self.counter_train = 0
        self._sum_writer = None
        self._merged = None
        self._init_model()

    def dqn_network(self, input_size, scope_name, act=tf.nn.relu):

        with tf.variable_scope(scope_name):
            hidden1 = tf.layers.dense(input_size, self.n_hidden_1, activation=act, name=scope_name + "_1",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, self.n_hidden_2, activation=act, name=scope_name + "_2",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden3 = tf.layers.dense(hidden2, self.n_hidden_3, activation=act, name=scope_name + "_3",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

            prediction = tf.layers.dense(hidden3, self.output_shape)

        return prediction

    def _init_model(self):
        with tf.variable_scope("DoubleDQN_Input_Data"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="state")
            self._y = tf.placeholder("float", [None, self.output_shape], name="output")

        self._prediction = self.dqn_network(input_size=self._x, scope_name="Double_Q_Primary")
        self._target = self.dqn_network(input_size=self._x, scope_name="Double_Q_Target")

        with tf.variable_scope("DoubleDQN_Learning"):
            self._loss = tf.losses.mean_squared_error(self._y, self._prediction)
            # self._loss = tf.losses.huber_loss(self._y, self._prediction)
            self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)

        summary = tf.summary.scalar('loss_doubledqn', self._loss)

        self._merged = tf.summary.merge([summary])
        self._sum_writer = tf.summary.FileWriter(self.path, self.session.graph)

        self.saver = tf.train.Saver()
        # self.saver = tf.train.Saver({'Double_Q_Primary': self._prediction, 'Double_Q_Target': self._target})

    def update_from_experience(self):
        # Randomly sample from experience
        minibatch = random.sample(self.memory, self.batch_size)
        # Initialize x and y for the neural network
        x = np.zeros((self.batch_size, self.input_shape))
        y = np.zeros((self.batch_size, self.output_shape))

        # Iterate over the minibatch to fill x and y
        i = 0
        for ss, aa, rr, ss_prime in minibatch:
            # x is simply the state
            x[i] = ss
            # y are the q values for each action.
            y[i] = self.predict(ss)

            # We update the action taken ONLY.
            if ss_prime is not None:
                # ss_prime is not None, so this is not a terminal state.
                q_sa = self.predict(ss_prime)
                idx = np.argmax(q_sa)
                q_target = self.predict_target(ss_prime)[0][idx]
                y[i, aa] = rr / 10 + self.gamma * q_target
            else:
                # ss_prime is None so this is a terminal state.
                y[i, aa] = rr / 10
            i += 1

        self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        self.counter_train += 1

        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        summary, _, loss = self.session.run([self._merged, self._optimizer, self._loss], feed_dict)

        self._sum_writer.add_summary(summary, self.counter_train)

        if self.counter_train % self.save_update == 0:
            self.save()

    def predict(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self.session.run(self._prediction, feed_dict)

    def predict_target(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self.session.run(self._target, feed_dict)

    def update_target(self):
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Primary")
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Target")

        assert len(q_vars) == len(q_target_vars)

        # hard update
        self.session.run([v_t.assign(v) for v_t, v in zip(q_target_vars, q_vars)])

    def load(self, name="model-doubledqn"):
        save_path = self.saver.restore(self.session, self.path + "/models/model.ckpt")

    def name(self):
        return "Double DQN"


class DuelingDQNEstimator(ValueEstimator):
    n_hidden_1 = 256
    n_hidden_2 = 512

    def __init__(self, session, input_shape, output_shape=Card.DIFFERENT_CARDS, memory=100000, batch_size=1024,
                 gamma=0.95, path="log", save_update=5000):
        super(DuelingDQNEstimator, self).__init__(session, path, input_shape, output_shape, memory, batch_size, False)
        self.logger = logging.getLogger('wizard-rl.DuelingDQNEstimator')

        self.gamma = gamma
        self.learning_rate = 0.001
        self.save_update = save_update
        self.t_train = 0

        self._state = None
        self._y = None
        self.q_values = None

        self._optimizer = None
        self._merged = None
        self._sum_writer = None

        self._init_model()

    def _init_model(self):
        with tf.variable_scope("DuelingDQN_Input"):
            self._state = tf.placeholder("float", [None, self.input_shape], name="State")
            self._y = tf.placeholder("float",  [None, self.output_shape], name="Q-Values")

        with tf.variable_scope("Dueling_Network"):
            hidden1_v = tf.layers.dense(self._state, self.n_hidden_1)
            hidden2_v = tf.layers.dense(hidden1_v, self.n_hidden_2)

            value = tf.layers.dense(hidden2_v, 1)

            hidden1_a = tf.layers.dense(self._state, self.n_hidden_1)
            hidden2_a = tf.layers.dense(hidden1_a, self.n_hidden_2)

            advantage = tf.layers.dense(hidden2_a, self.output_shape)

            self.q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
            # self.q_values = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

        with tf.variable_scope("DuelingDQN_Learning"):
            self._loss = tf.losses.mean_squared_error(self._y, self.q_values)
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

        summary = tf.summary.scalar('loss_dueling', self._loss)
        self._merged = tf.summary.merge([summary])
        self._sum_writer = tf.summary.FileWriter(self.path , self.session.graph)

        self.saver = tf.train.Saver()
        # self.saver = tf.train.Saver({'q-values-dueling':self.q_values})

    def update_from_experience(self):
        minibatch = random.sample(self.memory, self.batch_size)

        # Initialize x and y for the neural network
        x = np.zeros((self.batch_size, self.input_shape))
        y = np.zeros((self.batch_size, self.output_shape))

        # Iterate over the minibatch to fill x and y
        i = 0
        for ss, aa, rr, ss_prime in minibatch:
            # x is simply the state
            x[i] = ss
            # y are the q values for each action.
            y[i] = self.predict(ss)

            # We update the action taken ONLY.
            if ss_prime is not None:
                # ss_prime is not None, so this is not a terminal state.
                q_sa = self.predict(ss_prime)
                y[i, aa] = rr / 10 + np.max(q_sa)
            else:
                # ss_prime is None so this is a terminal state.
                y[i, aa] = rr / 10
            i += 1

        self.train_model(x, y)

    def train_model(self, batch_x, batch_y):
        self.t_train += 1

        feed_dict = {
            self._state: batch_x,
            self._y: batch_y
        }
        # train network
        summary, _, loss = self.session.run([self._merged, self._optimizer, self._loss], feed_dict)

        self._sum_writer.add_summary(summary, self.t_train)

        if self.t_train % self.save_update == 0:
            self.save()

    def update_target(self):
        pass

    def predict(self, s):
        feed_dict = {self._state: np.array(s)[np.newaxis, :]}
        return self.session.run(self.q_values, feed_dict)

    def name(self):
        return "Dueling DQN"

    def load(self):
        self.saver.restore(self.session, self.path + "/models/model.ckpt")


