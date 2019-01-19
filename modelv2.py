import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import math

from tensorlayer.prepro import *
from tensorlayer.layers import *

class network_model:

    def __init__(self, n_pkmn, latent_dim, optimizer):

        self.ph_X1 = tf.placeholder('int32', [None, 6], name = "p1pkmn")
        self.ph_X2 = tf.placeholder('int32', [None, 6], name = "p2pkmn")
        self.ph_gtY = tf.placeholder('float32', [None, 2])

        self.latent_dim = latent_dim
        self.n_pkmn = n_pkmn
        self.optimizer = optimizer

        self.combine_method = tf.reduce_mean
        self.l1 = 0.0
        self.l2 = 0.0

    # Pokemon network
    def pkmn_model(self, input_data, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=0.5)
        b_init = tf.random_normal_initializer(stddev=0.5)
        g_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

        with tf.variable_scope("pkmn_network", reuse=reuse) as vs:
            
            n = InputLayer(input_data, name = 'input')
            
            # parallel fcs as 1dconv
            n = Conv1d(n, n_filter = 128, filter_size = 1, W_init = w_init, b_init = b_init, name = 'fc1')
            n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

            n = Conv1d(n, n_filter = output_dim, filter_size = 1, W_init = w_init, b_init = b_init, name = 'fc2')
            n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")
            # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout2")
        
        return n

    # Battle network
    def battle_model(self, input_data, is_train = False, reuse = False):
        
        w_init = tf.random_normal_initializer(stddev=0.5)
        b_init = tf.random_normal_initializer(stddev=0.5)
        g_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

        with tf.variable_scope("battle_network", reuse=reuse) as vs:
            
            n = InputLayer(input_data, name = 'input')
            
            n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
            n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")
            # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout1")

            n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
            n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")
            n = DropoutLayer(n, keep = 0.5, is_fix = True, is_train = is_train, name = "dropout2")

            logits = DenseLayer(n, 2, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc4")
        
        return logits

    def build_model(self, is_train, reuse):

        x1_onehot = tf.one_hot(self.ph_X1, self.n_pkmn)
        x2_onehot = tf.one_hot(self.ph_X2, self.n_pkmn)

        p1pkmn_latents = self.pkmn_model(x1_onehot, self.latent_dim, is_train = is_train, reuse = reuse).outputs
        p2pkmn_latents = self.pkmn_model(x2_onehot, self.latent_dim, is_train = is_train, reuse = True).outputs

        p1team = self.combine_method(p1pkmn_latents, axis = 1)
        p2team = self.combine_method(p2pkmn_latents, axis = 1)
        battlefield = tf.concat([p1team, p2team], axis = 1)

        predict_logits = self.battle_model(battlefield, is_train = is_train, reuse = reuse).outputs
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = predict_logits, labels = self.ph_gtY))
        
        predict_softmax = tf.nn.softmax(predict_logits)
        prediction = tf.argmax(predict_logits, 1)
        correct_mat = tf.equal(tf.argmax(self.ph_gtY, 1), prediction)
        acc = tf.reduce_mean(tf.cast(correct_mat, tf.float32))

        return loss, acc, predict_softmax, prediction

    def init(self):

        self.raw_train_loss, self.train_acc, _, _ = self.build_model(True, False)
        self.val_loss, self.val_acc, self.val_predictprob, self.val_prediction = self.build_model(False, True)

        reg_loss = 0
        trainable_vars = tf.trainable_variables()

        if self.l1 > 0:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale = self.l1, scope = None)
            reg_loss += tf.contrib.layers.apply_regularization(l1_regularizer, trainable_vars)
        
        if self.l2 > 0:
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale = self.l2, scope = None)
            reg_loss += tf.contrib.layers.apply_regularization(l2_regularizer, trainable_vars)
        
        # Orthogonal regularization?
        # Spectral?

        self.train_loss = self.raw_train_loss + reg_loss

        self.train_op = self.optimizer.minimize(self.train_loss)
