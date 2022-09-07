import json
import unicodedata
import re
from voc import *

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