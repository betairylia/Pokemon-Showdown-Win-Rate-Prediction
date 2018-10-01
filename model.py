import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import math

from tensorlayer.prepro import *
from tensorlayer.layers import *

# RNN Cell
def input_rnn_cell(n_hidden_units):

    cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_hidden_units, state_is_tuple = True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.7)

    return cell

# Pokemon network
def pkmn_model(input_data, output_dim, is_train = False, reuse = False):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("pkmn_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")

        n = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3")
        n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")
    
    return n

# Team network
def team_model(input_data, output_dim, is_train = False, reuse = False):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("team_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")

        n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3")
        n = DropoutLayer(n, keep = 0.8, is_fix = True, is_train = is_train, name = "dropout3")

        n = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc4")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn4")
        n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout4")
    
    return n

# Battle network
def battle_model(input_data, is_train = False, reuse = False):
    
    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("battle_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")
        n = DropoutLayer(n, keep = 0.8, is_fix = True, is_train = is_train, name = "dropout2")

        n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3")
        n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")

        logits = DenseLayer(n, 2, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc4")
    
    return logits

# Battle network (-big)
def big_model(input_data, is_train = False, reuse = False):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("big_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        n = DenseLayer(n, 1024, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")
        n = DropoutLayer(n, keep = 0.8, is_fix = True, is_train = is_train, name = "dropout2")

        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3")
        n = DropoutLayer(n, keep = 0.8, is_fix = True, is_train = is_train, name = "dropout3")

        n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc4")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn4")
        n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout4")

        logits = DenseLayer(n, 2, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc5")

    return logits
