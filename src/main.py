import logging
import tensorflow as tf
import os

from Training.TrainingMethods import SelfPlayTraining, TrainingAgainstOtherPlayer
from Training.TrainingMethods import DUELING_PLAYER, PG_PLAYER, DDQN_PLAYER, DQN_PLAYER


def init_logger(console_logging=False, console_level=logging.ERROR):
    # create logger with 'wizard-rl'
    logger = logging.getLogger('wizard-rl')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs debug messages
    path = "log/"
    filename = path + "wizard-rl.log"
    if not os.path.exists(path):
        os.makedirs(path)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(fh)

    if console_logging:
        # create console handler with higher log level
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def others_training_agent(type, tp, train_rounds, interval):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                          intra_op_parallelism_threads=2,
                                          use_per_session_threads=True)) as sess:
        training = TrainingAgainstOtherPlayer(session=sess, players_type=type, num_games=train_rounds, tp=tp,
                                              interval=interval)
        sess.run(tf.global_variables_initializer())
        training.train_agent()


def others_training_ddqn(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    others_training_agent(DDQN_PLAYER, tp, train_rounds, interval, evaluation_interval)


def others_training_dqn(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    others_training_agent(DQN_PLAYER, tp, train_rounds, interval, evaluation_interval)


def others_training_dueling(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    others_training_agent(DUELING_PLAYER, tp, train_rounds, interval, evaluation_interval)


def others_training_pg(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    others_training_agent(PG_PLAYER, tp, train_rounds, interval, evaluation_interval)


def self_training_agent(type, tp, train_rounds, interval, evaluation_interval):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                          intra_op_parallelism_threads=2,
                                          use_per_session_threads=True)) as sess:
        training = SelfPlayTraining(session=sess, players_type=type, num_games=train_rounds, tp=tp, interval=interval)
        sess.run(tf.global_variables_initializer())
        training.train_agent(evaluation_interval)


def self_training_ddqn(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    self_training_agent(DDQN_PLAYER, tp, train_rounds, interval, evaluation_interval)


def self_training_dueling(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    self_training_agent(DUELING_PLAYER, tp, train_rounds, interval, evaluation_interval)


def self_training_dqn(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    self_training_agent(DQN_PLAYER, tp, train_rounds, interval, evaluation_interval)


def self_training_pg(tp=False, train_rounds=10000, interval=500, evaluation_interval=500):
    self_training_agent(PG_PLAYER, tp, train_rounds, interval, evaluation_interval)


if __name__ == "__main__":
    init_logger(console_logging=True, console_level=logging.DEBUG)

    self_training_dqn(tp=True)
    self_training_dqn(tp=False)
    self_training_pg(tp=True)
    self_training_pg(tp=False)
    self_training_ddqn(tp=True)
    self_training_ddqn(tp=False)
