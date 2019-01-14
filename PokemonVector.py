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

parser.add_argument('pokemonpath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-bs', '--batch-size', type = int, default = 16)
parser.add_argument('-maxl', '--name-max-length', type = int, default = 20, help = "Max length of pokemon name")
parser.add_argument('-hu', '--hidden-units', type = int, default = 128, help = "Hidden state vec length for RNN")
parser.add_argument('-pu', '--pkmn-units', type = int, default = 128, help = "Hidden state vec length for each pokemon")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")

args = parser.parse_args()

# Hyper parameters
batch_size = args.batch_size                    # batch size
n_inputs = 128                                  # ascii
n_steps = args.name_max_length                  # name length
n_hidden_units = args.hidden_units              # hidden state tensor length
n_pkmn_units = args.pkmn_units                  # pkmn state tensor length

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus
continue_loop = True

# Read the data
file_content = [line.rstrip('\n') for line in open(args.pokemonpath, 'r')]

dataCount = len(file_content)
pokemons = np.zeros((dataCount, n_steps))
pokemons_size = np.zeros((dataCount))
pokemon_vectors = np.zeros((dataCount, n_pkmn_units))

for i in range(len(file_content)):
    name_vec = np.array(np.array(file_content[i][:n_steps], 'c').view(np.uint8))
    name_len = name_vec.shape[0]
    pokemons_size[i] = name_len
    pokemons[i] = np.pad(name_vec, (0, n_steps - name_len), 'constant')

ph_pkmn = tf.placeholder('uint8', [None, n_steps], name = "p1pkmn")
ph_pkmn_size = tf.placeholder('int32', [None], name = "p1pkmn_size")

# cell = tf.contrib.rnn.BasicRNNCell(n_hidden_units)
cell = input_rnn_cell(n_hidden_units)
zero_state = cell.zero_state(batch_size, dtype = tf.float32)

# Collect pokemon latent vector
with tf.variable_scope("Pokemon_encode_RNN", reuse = tf.AUTO_REUSE):
    _, latent_pkmn_tmp = tf.nn.dynamic_rnn(
        cell = cell, 
        dtype = tf.float32,
        sequence_length = ph_pkmn_size,
        inputs = tf.one_hot(ph_pkmn[:, :], 128),
        initial_state = zero_state)

latent_pkmn = pkmn_model(latent_pkmn_tmp[1], n_pkmn_units, is_train = False, reuse = tf.AUTO_REUSE).outputs

# Create session
saver = tf.train.Saver()

sess = tf.Session()
# sess.run(tf.local_variables_initializer())
# tl.layers.initialize_global_variables(sess)

if args.load != "None":
    saver.restore(sess, args.load)
    print("Model loaded")

# Collect pokemon vectors
batch_count = math.ceil(dataCount / batch_size)

print("Collecting pokemon vectors...")

for i in range(batch_count):

    # Collect pokemon names
    batch_pkmn = np.zeros((batch_size, n_steps))
    batch_pkmn_size = np.zeros((batch_size))

    # Calculate actual batch size
    batch_start = i * batch_size
    batch_size_real = batch_size

    if batch_start + batch_size > dataCount:
        batch_size_real = dataCount - i * batch_size

    # Feed data
    batch_pkmn[0 : batch_size_real, :] = pokemons[batch_start : batch_start + batch_size_real]
    batch_pkmn_size[0 : batch_size_real] = pokemons_size[batch_start : batch_start + batch_size_real]

    # Run the model
    result = sess.run(latent_pkmn, feed_dict = {ph_pkmn : batch_pkmn, ph_pkmn_size : batch_pkmn_size})
    pokemon_vectors[batch_start : batch_start + batch_size_real] = result[0 : batch_size_real]

    print("%d / %d" % (batch_start + batch_size_real, dataCount))

print("Complete.")

def distance(a, b):
    return scipy.spatial.distance.euclidean(a, b)

def cosine_distance(a, b):
    return scipy.spatial.distance.cosine(a, b)

def by_distance(a):
    return a['dist']

def by_cosine_distance(a):
    return a['cosine_dist']

while continue_loop:

    pkmn_str = input("Please enter a pokemon name: ")

    target_id = -1
    for i in range(dataCount):
        if file_content[i] == pkmn_str:
            target_id = i
    
    if target_id == -1:
        print("Pokemon %s not fond in %s" % (pkmn_str, args.pokemonpath))

    pokemon_dict = []
    for i in range(dataCount):
        tmp = {}
        tmp['idx'] = i
        tmp['dist'] = distance(pokemon_vectors[target_id], pokemon_vectors[i])
        tmp['cosine_dist'] = cosine_distance(pokemon_vectors[target_id], pokemon_vectors[i])
        pokemon_dict.append(tmp)
    
    # Sort by Distance
    pokemon_dict.sort(key = by_distance)

    print("\n Sort by Distance: \n====================================\nIndex\tDist\tName")
    
    # The first one must be itself so start from 2nd
    for i in range(1, 16):
        print("%02d\t%02.2f\t%s" % (i, pokemon_dict[i]['dist'], file_content[pokemon_dict[i]['idx']]))

    # Sort by cosine distance
    pokemon_dict.sort(key = by_cosine_distance)

    print("\n Sort by Cosine Distance: \n====================================\nIndex\tcDist\tName")
    
    # The first one must be itself so start from 2nd
    for i in range(1, 16):
        print("%02d\t%02.2f\t%s" % (i, pokemon_dict[i]['cosine_dist'], file_content[pokemon_dict[i]['idx']]))

    print("\n Copyable:\n[LIST=1]")
    # The first one must be itself so start from 2nd
    for i in range(1, 13):
        print("[*]%01.2f  %s" % (pokemon_dict[i]['cosine_dist'], file_content[pokemon_dict[i]['idx']]))
    print("[/LIST]")
    
    print("\n\n")
