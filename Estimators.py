import random
import numpy as np
import tensorflow as tf
from Card import Card


class Estimator(object):
    """
    An state-action value function estimator. All state inputs must already
    be transformed by a 'featurizer' before being used with Estimator methods.
    """

    def __init__(self):
        self.model = None

    def update(self, s, a, r, s_prime):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class.")

    def predict(self, s):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def save(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")

    def load(self, name):
        raise NotImplementedError("This method must be implemented by"
                                  "your Estimator class")


class DQNEstimator(Estimator):
    n_hidden_1 = 256
    n_hidden_2 = 512
    n_hidden_3 = 1024

    def __init__(self, session, input_shape, output_shape=Card.DIFFERENT_CARDS, memory=100000, batch_size=1024,
                 gamma=0.95, target_update=5000, save_update=100000):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma
        self.memory = [([], 0, 0, [])] * memory
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_rate = max(1, batch_size // 8)
        self.save_update = save_update
        self.t = 0
        self._prediction = None
        self._optimizer = None
        self._loss = None
        self._x = None
        self._y = None
        self._target = None
        self.counter_train = 0
        self._var_init = None
        self._session = session
        self._sum_writer = None
        self._merged = None
        self._init_model()

    def dqn_network(self, input, scope_name, act=tf.nn.relu):

        with tf.variable_scope(scope_name):
            hidden1 = tf.layers.dense(input, self.n_hidden_1, activation=act, name=scope_name + "_1",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, self.n_hidden_2, activation=act, name=scope_name + "_2",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden3 = tf.layers.dense(hidden2, self.n_hidden_3, activation=act, name=scope_name + "_3",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

            prediction = tf.layers.dense(hidden3, self.output_shape)

        return prediction

    def _init_model(self):
        with tf.variable_scope("Input_Data_CardPrediction"):
            self._x = tf.placeholder("float", [None, self.input_shape], name="state")
            self._y = tf.placeholder("float", [None, self.output_shape], name="output")

        self._prediction = self.dqn_network(input=self._x, scope_name="Q_Primary")
        self._target = self.dqn_network(input=self._x, scope_name="Q_Target")

        with tf.variable_scope("Learning_CardPrediction"):
            self._loss = tf.losses.mean_squared_error(self._y, self._prediction)
            self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)

        sum_loss = tf.summary.scalar('loss_card-prediction', self._loss)

        self._merged = tf.summary.merge([sum_loss])
        self._sum_writer = tf.summary.FileWriter("log/dqn/train-summary", self._session.graph)
        self._var_init = tf.global_variables_initializer()

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

        if self.t >= len(self.memory) and self.t % self.target_update == 0:
            self.update_target()
        # If memory is full, we can start training
        if self.t >= len(self.memory) and self.t % self.update_rate == 0:
            # Randomly sample from experience
            minibatch = random.sample(self.memory, self.batch_size)
            # Initialize x and y for the neural network
            x = np.zeros((self.batch_size, len(s)))
            y = np.zeros((self.batch_size, Card.DIFFERENT_CARDS))

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
                    q_sa = self.predict_target(ss_prime)
                    y[i, aa] = rr / 10 + self.gamma * np.max(q_sa)
                else:
                    # ss_prime is None so this is a terminal state.
                    y[i, aa] = rr / 10
                i += 1

            self.train_model(x, y)

        # save model after a given number of steps (save_update)
        # if self.t % self.save_update == 0:
        #    self.save('dqn_ckpt_' + self.t)

    def train_model(self, batch_x, batch_y):
        self.counter_train += 1

        feed_dict = {
            self._x: batch_x,
            self._y: batch_y
        }
        # train network
        summary, _, loss = self._session.run([self._merged, self._optimizer, self._loss], feed_dict)

        self._sum_writer.add_summary(summary, self.counter_train)

    def predict(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self._session.run(self._prediction, feed_dict)

    def predict_target(self, s):
        feed_dict = {self._x: np.array(s)[np.newaxis, :]}
        return self._session.run(self._target, feed_dict)

    def update_target(self):
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Primary")
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Target")

        assert len(q_vars) == len(q_target_vars)

        # hard update
        self._session.run([v_t.assign(v) for v_t, v in zip(q_target_vars, q_vars)])

    def save(self, name="model-dqn"):
        print("Saving {}".format(name))
        # create saver
        saver = tf.train.Saver()

        path = saver.save(self.session, "/tmp/{}.ckpt".format(name))

        print("Saved in path {}".format(path))

    def load(self, name="model-dqn"):
        saver = tf.train.Saver()
        saver.restore(self.session, "/tmp/{}.ckpt".format(name))

    def default_weights(self, n_input, n_output):
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_3, n_output])),
        }

        return weights

    def default_biases(self, n_output):
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }

        return biases

    def close(self):
        self._session.close()
