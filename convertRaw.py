import math
import argparse
import random
import sys
import os

parser = argparse.ArgumentParser(description="Convert raw names to embedded ids")

parser.add_argument('datapath')
parser.add_argument('-max', '--max-species', type = int, default = 0, help = "Max species to be embedded, 0 = no limit. most N-1 used pokemons will be embedded into corresponding one-hot, other pokemons share a single Nth entry.")

args = parser.parse_args()

data_File = open(args.datapath, 'r')
data_fileContent = [line.rstrip('\n') for line in data_File]

prefix = args.datapath.split('.')[0]

out_file = open(prefix + '_train_emb.txt', 'w')
out_val_file = open(prefix + '_validation_emb.txt', 'w')
dict_file = open(prefix + '_dict.txt', 'w')

dataCount = len(data_fileContent) // 4

pkmn_dict = {}
pkmn_cnt = 0

for i in range(len(data_fileContent) // 4):
    
    p1pkmn = data_fileContent[i * 4 + 1].split(',')[:-1]
    p2pkmn = data_fileContent[i * 4 + 2].split(',')[:-1]

    p1pkmn_id = []
    p2pkmn_id = []

    for j in range(6):

        if p1pkmn[j] not in pkmn_dict:
            pkmn_cnt += 1
            pkmn_dict[p1pkmn[j]] = 0
        
        if p2pkmn[j] not in pkmn_dict:
            pkmn_cnt += 1
            pkmn_dict[p2pkmn[j]] = 0
        
        # Count usage
        pkmn_dict[p1pkmn[j]] += 1
        pkmn_dict[p2pkmn[j]] += 1

# Sort dict ( more used -> never used )
sorted_entries = [(k, pkmn_dict[k]) for k in sorted(pkmn_dict, key = pkmn_dict.get, reverse = True)]

# Assign entries
i = 0
for k, v in sorted_entries:
    pkmn_dict[k] = (min(i, args.max_species - 1), v)
    i += 1

for i in range(len(data_fileContent) // 4):

    p1pkmn = data_fileContent[i * 4 + 1].split(',')[:-1]
    p2pkmn = data_fileContent[i * 4 + 2].split(',')[:-1]

    p1pkmn_id = []
    p2pkmn_id = []

    for j in range(6):

        p1pkmn_id.append(str(pkmn_dict[p1pkmn[j]][0]))
        p2pkmn_id.append(str(pkmn_dict[p2pkmn[j]][0]))
    
    if random.random() < 0.2:
        out_val_file.write("%s\n%s\n%s\n%s\n" % (\
            data_fileContent[i * 4],\
            ','.join(p1pkmn_id),\
            ','.join(p2pkmn_id),\
            data_fileContent[i * 4 + 3]))
    else:
        out_file.write("%s\n%s\n%s\n%s\n" % (\
            data_fileContent[i * 4],\
            ','.join(p1pkmn_id),\
            ','.join(p2pkmn_id),\
            data_fileContent[i * 4 + 3]))

dict_file.write("%d\n" % min(args.max_species, pkmn_cnt))
for k, v in sorted_entries:
    dict_file.write("%d,%s\n" % (pkmn_dict[k][0], k))

out_file.close()
dict_file.close()
