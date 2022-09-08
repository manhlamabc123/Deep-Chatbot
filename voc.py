PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-Of-Sentence token
EOS_token = 2 # End-Of-Sentence token

# This class keeps a mapping from words to indexes, a reverse mapping of indexes to words, a count of each word and a total word count.
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3 # Count SOS, EOS, PAD

    # Adding all words in a sentence
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    # Adding a word to the vocabulary
    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_words] = word
            self.num_words += 1
        else:
            self.word_to_count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word_to_count.items():
            if v >= min_count:
                keep_words.append(k)

        print(f"keep_words {len(keep_words)} / {len(self.word_to_index)} = {len(keep_words) / len(self.word_to_index):.4f}")

        # Reinitialize dictionaries
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)