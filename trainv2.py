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

from modelv2 import network_model as network_model

from tensorlayer.prepro import *
from tensorlayer.layers import *
from termcolor import colored, cprint

parser = argparse.ArgumentParser(description="Run the NN for pokemon showdown replays")

parser.add_argument('dataprefix')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epoches', type = int, default = 200)
parser.add_argument('-bs', '--batch-size', type = int, default = 64)
parser.add_argument('-zdim', '--latent-dim', type = int, default = 128, help = "Hidden state vec length for each pokemon")
parser.add_argument('-fmt', '--format', type = str, default = "Unknown", help = 'Format for the data, used to store models')
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-log', '--logpath', type = str, default = "logs", help = "Tensorboard log file output path")
parser.add_argument('-name', '--logname', type = str, default = "", help = "Tensorboard log file output name")
parser.add_argument('-maxpool', '--maxpool', dest = "combine_method", action = "store_const", default = tf.reduce_mean, const = tf.reduce_max, help = "Use max pooling instead of summation to combine pokemons to a team")
parser.add_argument('-l1reg', '--l1-regularization', type = float, default = 0.0, help = "Coefficient for L1 regularization over all trainable variables")
parser.add_argument('-l2reg', '--l2-regularization', type = float, default = 0.0, help = "Coefficient for L2 regularization over all trainable variables")

args = parser.parse_args()

# Hyper parameters
total_epoches = args.epoches                    # epoches
batch_size = args.batch_size                    # batch size

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

logpath = os.path.join(args.logpath, args.logname + '-' + args.dataprefix + '/')

train_File = open(args.dataprefix + '_train_emb.txt', 'r')
train_fileContent = [line.rstrip('\n') for line in train_File]

validation_File = open(args.dataprefix + '_validation_emb.txt', 'r')
validation_fileContent = [line.rstrip('\n') for line in validation_File]

dict_File = open(args.dataprefix + '_dict.txt', 'r')
n_pkmn = int(dict_File.readline())

def get_batch(fileContent, batchsize):

    batch_X1 = np.zeros((batchsize, 6), np.int32)
    batch_X2 = np.zeros((batchsize, 6), np.int32)
    batch_Y  = np.zeros((batchsize, 2))

    file_len = len(fileContent) // 4
    
    replay_order = list(range(file_len))
    random.shuffle(replay_order)
    
    batch_cnt = 0

    for i in replay_order:

        p1_pkmn_list = fileContent[i * 4 + 1].split(',')
        p2_pkmn_list = fileContent[i * 4 + 2].split(',')
        winner = int(fileContent[i * 4 + 3]) - 1

        # pkmn list
        batch_X1[batch_cnt, :] = np.asarray([int(s) for s in p1_pkmn_list])
        batch_X2[batch_cnt, :] = np.asarray([int(s) for s in p2_pkmn_list])

        # winner id to one-hot
        batch_Y[batch_cnt] = np.eye(2)[winner]

        batch_cnt += 1
        if batch_cnt >= batchsize:
            batch_cnt = 0
            yield batch_X1, batch_X2, batch_Y

def get_epoch():
    for i in range(total_epoches):
        yield get_batch(train_fileContent, args.batch_size), get_batch(validation_fileContent, args.batch_size)

optimizer = tf.train.AdamOptimizer()
model = network_model(n_pkmn, args.latent_dim, optimizer)

model.combine_method = args.combine_method
model.l1 = args.l1_regularization
model.l2 = args.l2_regularization

model.init()

train_loss = tf.summary.scalar('Training Loss', model.train_loss)
train_acc = tf.summary.scalar('Training Acc', model.train_acc)
val_loss = tf.summary.scalar('Validation Loss', model.val_loss)
val_acc = tf.summary.scalar('Validation Acc', model.val_acc)

merge_train = tf.summary.merge([train_loss, train_acc])
merge_validation = tf.summary.merge([val_loss, val_acc])

sess = tf.Session()
sess.run(tf.local_variables_initializer())
tl.layers.initialize_global_variables(sess)

train_writer = tf.summary.FileWriter(logpath + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(logpath + '/validation', sess.graph)

# TODO: save & load

train_iter = 0
val_iter = 0
epoch_idx = 0

for epoch_train, epoch_val in get_epoch():
    for _x1, _x2, _y in epoch_train:
        feed_dict = {model.ph_X1: _x1, model.ph_X2: _x2, model.ph_gtY: _y}
        _, n_loss, acc, summary = sess.run([model.train_op, model.train_loss, model.train_acc, merge_train], feed_dict = feed_dict)

        print(colored('Ep %03d - Train iter %06d' % (epoch_idx, train_iter), 'yellow') + colored('\tLoss = %04.4f, Acc = %.4f' % (n_loss, acc), 'green'))
        train_writer.add_summary(summary, train_iter)
        train_iter += 1

    for _x1, _x2, _y in epoch_val:
        feed_dict = {model.ph_X1: _x1, model.ph_X2: _x2, model.ph_gtY: _y}
        n_loss, acc, summary = sess.run([model.val_loss, model.val_acc, merge_validation], feed_dict = feed_dict)

        print(colored('Ep %03d -   Val iter %06d' % (epoch_idx, val_iter), 'yellow') + colored('\tLoss = %04.4f, Acc = %.4f' % (n_loss, acc), 'green'))
        validation_writer.add_summary(summary, val_iter)
        val_iter += 1
    epoch_idx += 1
