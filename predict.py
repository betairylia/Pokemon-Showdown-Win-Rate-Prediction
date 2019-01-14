import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import time
import math
import argparse
import random
import sys
import os
import matplotlib.pyplot as plt

from model import *

from tensorlayer.prepro import *
from tensorlayer.layers import *
from termcolor import colored, cprint

parser = argparse.ArgumentParser(description="Run the NN for pokemon showdown replays")

# parser.add_argument('trainpath')
# parser.add_argument('testpath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epoches', type = int, default = 200)
parser.add_argument('-bs', '--batch-size', type = int, default = 16)
parser.add_argument('-maxl', '--name-max-length', type = int, default = 20, help = "Max length of pokemon name")
parser.add_argument('-hu', '--hidden-units', type = int, default = 128, help = "Hidden state vec length for RNN")
parser.add_argument('-pu', '--pkmn-units', type = int, default = 128, help = "Hidden state vec length for each pokemon")
parser.add_argument('-thu', '--team-hidden-units', type = int, default = 192, help = "Hidden state vec length for teams")
parser.add_argument('-big', '--big-model', dest = "use_big_model", action = 'store_const', const = True, default = False, help = "Direct concat to battle instead calculate team vec")
parser.add_argument('-fmt', '--format', type = str, default = "Unknown", help = 'Format for the data, used to store models')
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")

args = parser.parse_args()

# Hyper parameters
total_epoches = args.epoches                    # epoches
batch_size = args.batch_size                    # batch size
n_inputs = 128                                  # ascii
n_steps = args.name_max_length                  # name length
n_hidden_units = args.hidden_units              # hidden state tensor length
n_pkmn_units = args.pkmn_units                  # pkmn state tensor length
n_team_vector_length = args.team_hidden_units   # Team tensor length
use_direct_concat = args.use_big_model          # Use a big network instead of two team first

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus
continue_loop = True

ph_p1pkmn = tf.placeholder('uint8', [None, 6, n_steps], name = "p1pkmn")
ph_p2pkmn = tf.placeholder('uint8', [None, 6, n_steps], name = "p2pkmn")
ph_p1pkmn_size = tf.placeholder('int32', [None, 6], name = "p1pkmn_size")
ph_p2pkmn_size = tf.placeholder('int32', [None, 6], name = "p2pkmn_size")

ph_winner = tf.placeholder('float32', [None, 2], name = "winner_gt")

# cell = tf.contrib.rnn.BasicRNNCell(n_hidden_units)
cell = input_rnn_cell(n_hidden_units)
zero_state = cell.zero_state(batch_size, dtype = tf.float32)

# Collect pokemon latent vector
latent_p1pkmn = []
latent_p2pkmn = []

# tst_pm0 = tf.one_hot(ph_p1pkmn[:, 0, :], 128)

for i in range(6):
    with tf.variable_scope("Pokemon_encode_RNN", reuse = tf.AUTO_REUSE):
        _, latent_p1pkmn_tmp = tf.nn.dynamic_rnn(
            cell = cell, 
            dtype = tf.float32,
            sequence_length = ph_p1pkmn_size[:, i],
            inputs = tf.one_hot(ph_p1pkmn[:, i, :], 128),
            initial_state = zero_state)

        _, latent_p2pkmn_tmp = tf.nn.dynamic_rnn(
            cell = cell, 
            dtype = tf.float32,
            sequence_length = ph_p2pkmn_size[:, i],
            inputs = tf.one_hot(ph_p2pkmn[:, i, :], 128),
            initial_state = zero_state)
    
    print(latent_p1pkmn_tmp)
    
    latent_p1pkmn.append(pkmn_model(latent_p1pkmn_tmp[1], n_pkmn_units, is_train = True, reuse = tf.AUTO_REUSE).outputs)
    latent_p2pkmn.append(pkmn_model(latent_p2pkmn_tmp[1], n_pkmn_units, is_train = True, reuse = tf.AUTO_REUSE).outputs)

# Build team vector
if(use_direct_concat == True):

    # Predict battle results
    battle_tensor = tf.concat(latent_p1pkmn + latent_p2pkmn, 1)

    print(battle_tensor.shape)

    net = big_model(battle_tensor, is_train = True, reuse = False)
    logits_train = net.outputs

else:

    team1_tensor = tf.concat(latent_p1pkmn, 1)
    team2_tensor = tf.concat(latent_p2pkmn, 1)

    print(latent_p1pkmn[0].shape)
    print(team1_tensor.shape)

    team1_net = team_model(team1_tensor, n_team_vector_length, is_train = True, reuse = False)
    team2_net = team_model(team2_tensor, n_team_vector_length, is_train = True, reuse = True)

    # Predict battle results
    battle_tensor = tf.concat([team1_net.outputs, team2_net.outputs], 1)

    print(battle_tensor.shape)

    net = battle_model(battle_tensor, is_train = True, reuse = False)
    logits_train = net.outputs

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_train, labels = ph_winner))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# Testing
# Collect pokemon latent vector
latent_p1pkmn_test = []
latent_p2pkmn_test = []
for i in range(6):

    with tf.variable_scope("Pokemon_encode_RNN", reuse = tf.AUTO_REUSE):
        _, latent_p1pkmn_tmp = tf.nn.dynamic_rnn(
            cell = cell, 
            dtype = tf.float32,
            sequence_length = ph_p1pkmn_size[:, i],
            inputs = tf.one_hot(ph_p1pkmn[:, i, :], 128),
            initial_state = zero_state)

        _, latent_p2pkmn_tmp = tf.nn.dynamic_rnn(
            cell = cell, 
            dtype = tf.float32,
            sequence_length = ph_p2pkmn_size[:, i],
            inputs = tf.one_hot(ph_p2pkmn[:, i, :], 128),
            initial_state = zero_state)

    latent_p1pkmn_test.append(pkmn_model(latent_p1pkmn_tmp[1], n_pkmn_units, is_train = False, reuse = tf.AUTO_REUSE).outputs)
    latent_p2pkmn_test.append(pkmn_model(latent_p2pkmn_tmp[1], n_pkmn_units, is_train = False, reuse = tf.AUTO_REUSE).outputs)

if(use_direct_concat == True):
    
    # Predict battle results
    battle_tensor = tf.concat(latent_p1pkmn_test + latent_p2pkmn_test, 1)
    net_test = big_model(battle_tensor, is_train = False, reuse = True)
    logits_test = net_test.outputs

else:
    
    team1_tensor_test = tf.concat(latent_p1pkmn_test, 1)
    team2_tensor_test = tf.concat(latent_p2pkmn_test, 1)

    team1_net_test = team_model(team1_tensor_test, n_team_vector_length, is_train = False, reuse = True)
    team2_net_test = team_model(team2_tensor_test, n_team_vector_length, is_train = False, reuse = True)

    battle_tensor = tf.concat([team1_net_test.outputs, team2_net_test.outputs], 1)

    net_test = battle_model(battle_tensor, is_train = False, reuse = True)
    logits_test = net_test.outputs

predict_softmax = tf.nn.softmax(logits_test)
acc, acc_op = tf.metrics.accuracy(labels = tf.argmax(ph_winner, 1), predictions = tf.argmax(logits_test, 1))

# Create session
saver = tf.train.Saver()

sess = tf.Session()
# sess.run(tf.local_variables_initializer())
# tl.layers.initialize_global_variables(sess)

if args.load != "None":
    saver.restore(sess, args.load)
    print("Model loaded")

while continue_loop:

    pkmnp1_str = input("Please enter player1 pokemon: ")
    pkmnp2_str = input("Please enter player2 pokemon: ")

    pkmnp1 = np.zeros((batch_size, 6, n_steps))
    pkmn_name_len_p1 = np.zeros((batch_size, 6))
    pkmnp2 = np.zeros((batch_size, 6, n_steps))
    pkmn_name_len_p2 = np.zeros((batch_size, 6))

    i = 0
    for j in range(6):

        s = np.array(np.array(pkmnp1_str.split(',')[:-1][j][:n_steps], 'c').view(np.uint8)).shape[0]
        pkmn_name_len_p1[i, j] = s
        pkmnp1[i, j] = np.pad(np.array(np.array(pkmnp1_str.split(',')[:-1][j][:n_steps], 'c').view(np.uint8)), (0, n_steps - s), 'constant')

        s = np.array(np.array(pkmnp2_str.split(',')[:-1][j][:n_steps], 'c').view(np.uint8)).shape[0]
        pkmn_name_len_p2[i, j] = s
        pkmnp2[i, j] = np.pad(np.array(np.array(pkmnp2_str.split(',')[:-1][j][:n_steps], 'c').view(np.uint8)), (0, n_steps - s), 'constant')
    
    # print(pkmnp1)
    # print(pkmnp2)

    feed_dict = {
        ph_p1pkmn: pkmnp1, ph_p2pkmn: pkmnp2,
        ph_p1pkmn_size: pkmn_name_len_p1, ph_p2pkmn_size: pkmn_name_len_p2}
            
    # dp_dict = tl.utils.dict_to_one( net.all_drop )
    # feed_dict.update( dp_dict )

    predict_result = sess.run(predict_softmax, feed_dict = feed_dict)

    print("Player 1 win rate: %.6f" % predict_result[0, 0])
    print("Player 2 win rate: %.6f" % predict_result[0, 1])
