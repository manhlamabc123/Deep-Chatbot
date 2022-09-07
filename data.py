import json

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
