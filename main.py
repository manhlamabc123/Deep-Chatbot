from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
from data import *


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Load & Preprocess Data
corpus_name = 'movie-corpus'
corpus = os.path.join('data', corpus_name)

# Create formatted data file
## Define path to new file
datafile = os.path.join(corpus, 'formatted_movie_lines.txt')
delimiter = '\t'

## Unescape the delimiter
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

## Initalize lines dict and conversations dict
lines = {}
conversations = {}

## Load lines and conversations
print("\nProcessing corpus into lines and conversations...")
lines, conversations = load_lines_and_conversations(os.path.join(corpus, 'utterances.jsonl'))

## Write new csv file
print("\nWritting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as output_file:
    writer = csv.writer(output_file, delimiter=delimiter, lineterminator='\n')
    for pair in extract_sentence_pairs(conversations):
        writer.writerow(pair)

# Load/Assemble Voc and pairs
save_dir = os.path.join('data', 'save')
voc, pairs = load_prepare_data(corpus, corpus_name, datafile, save_dir)

# Trim Voc and pairs
pairs = trim_rare_words(voc, pairs)

# Prepare data for models
small_batch_size = 5
batches = batch_to_train_data(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print('input_variable: ', input_variable)
print('lengths: ', lengths)
print('target_variable: ', target_variable)
print('mask: ', mask)
print('max_target_len: ', max_target_len)