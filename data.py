import itertools
import json
import unicodedata
import re
from voc import *
import torch

def print_lines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# Splits each line of the file to create lines and conversations
def load_lines_and_conversations(file_name):
    lines = {}
    conversations = {}
    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line_json = json.loads(line)
            # Extract fields for line object
            line_object = {}
            line_object['lineID'] = line_json['id']
            line_object['characterID'] = line_json['speaker']
            line_object['text'] = line_json['text']
            lines[line_object['lineID']] = line_object

            # Extract fields for conversation object
            if line_json['conversation_id'] not in conversations:
                conversation_object = {}
                conversation_object['conversationID'] = line_json['conversation_id']
                conversation_object['movieID'] = line_json['meta']['movie_id']
                conversation_object['lines'] = [line_object]
            else:
                conversation_object = conversations[line_json['conversation_id']]
                conversation_object['lines'].insert(0, line_object)
            conversations[conversation_object['conversationID']] = conversation_object
    return lines, conversations

# Extracts pairs of sentences from conversations
def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        # Iterate over all the lines of the conversation
        for i in range(len(conversation['lines']) - 1): # We ignore the last line (no answer for it)
            input_line = conversation['lines'][i]['text'].strip()
            target_line = conversation['lines'][i + 1]['text'].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs

# Preprocessing
MAX_LENGTH = 10 # Maximum sentence length to consider
MIN_COUNT = 3 # Minimum word count threshold for trimming

# Turn a Unicode string to plain ASCII
def unicode_to_ascii(string):
    return ''.join(
        character for character in unicodedata.normalize('NFD', string)
        if unicodedata.category(character) != 'Mn'
    )

# Lowercase, trim, and remove non-leter characters
def normalize_string(string):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string

# Read query/response pairs and return a Voc object
def read_voc(datafile, corpus_name):
    print('Reading lines...')
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(string) for string in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Return True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

# Filter pairs using filter_pair condition
def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

# Using the functions defined above, return a populated Voc object and pairs list
def load_prepare_data(corpus, corpus_name, datafile, save_dir):
    print('Start preparing training data...')
    voc, pairs = read_voc(datafile, corpus_name)
    print('Read {!s} sentence pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print('Counted words:', voc.num_words)
    return voc, pairs

def trim_rare_words(voc, pairs, MIN_COUNT=MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word_to_index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word_to_index:
                keep_output = False
                break
    
        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print(f"Trimmed from {len(pairs)} pairs to {len(keep_pairs)}, {len(keep_pairs) / len(pairs):.4f}")
    return keep_pairs

# Convert our English sentences to tensors by converting words to their indexes
def indexes_from_sentence(voc, sentence):
    return [voc.word_to_index[word] for word in sentence.split(' ')] + [EOS_token]

def zero_padding(line, fillvalue=PAD_token):
    return list(itertools.zip_longest(*line, fillvalue=fillvalue))

def binary_matrix(line, value=PAD_token):
    m = []
    for i, sequence in enumerate(line):
        m.append([])
        for token in sequence:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def input_var(line, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in line]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def output_var(line, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in line]
    max_target_len = max(len(indexes) for indexes in indexes_batch)
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

def batch_to_train_data(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)
    return inp, lengths, output, mask, max_target_len