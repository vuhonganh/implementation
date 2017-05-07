import numpy as np
import random


class TreebankReader:

    def __init__(self, path=None):
        if path is None:
            path = "data/stanfordSentimentTreebank"
        self._path = path
        self._sentences = None
        self._tokens = None
        self._vocab_size = None
    def get_sentences(self):
        """
        read the datasetSentences.txt and return a list of sentences, each sentence is a list of words
        """
        if self._sentences is not None:
            return self._sentences

        sentences = []
        with open(self._path + "/datasetSentences.txt", mode='r') as f:
            first = True  # skip first line
            for line in f:
                if first:
                    first = False
                    continue
                # strip white space at two ends and split to list of words, skip first word as it's index
                lw = line.strip().split()[1:]
                sentences.append([w.lower().decode('utf-8').encode('latin1') for w in lw])
        self._sentences = sentences
        return self._sentences

    def get_tokens(self):
        """
        return a dictionary token-idx and also compute a dictionary token-frequent 
        frequent is the number that token appears in this dataset
        """
        if self._tokens is not None:
            return self._tokens
        tokens = dict()
        id2Tokens = []
        ss = self.get_sentences()
        idx = 0
        for s in ss:
            for w in s:
                if w not in tokens:
                    tokens[w] = idx
                    id2Tokens.append(w)  # this list increase along with idx
                    idx += 1

        # add Unknown word
        tokens['UNK'] = idx
        self._tokens = tokens
        self._vocab_size = len(tokens)
        self._id2Tokens = id2Tokens
        return self._tokens

    def get_id2Tokens(self):
        if self._id2Tokens is None:
            self.get_tokens()

        return self._id2Tokens

    def get_vocab_size(self):
        if self._vocab_size is not None:
            return self._vocab_size
        self.get_tokens()
        return self._vocab_size

    def gen_batch(self, batch_size, window_size):
        """
        :param batch_size: number of input words  
        :param window_size: the maximum size around the input words
        :return: inputs, labels (note that one input can has up to 2*window_size labels
        """
        ss = self.get_sentences()  # get all sentences
        tk = self.get_tokens()  # get all tokens:
        nSentences = len(ss)
        # inputs = np.ndarray(shape=batch_size, dtype=np.int32)
        # labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        inputs = []
        labels = []
        i = 0
        while i < batch_size:
            id_sentence = random.randint(0, nSentences - 1)
            s = ss[id_sentence]
            nWords = len(s)  # length of sentence s
            if nWords > 1:
                center_id = random.randint(0, nWords - 1)
                center_word = s[center_id]
                context_words = s[max(0, center_id - window_size):min(nWords, center_id + window_size + 1)]
                real_context = [w for w in context_words if w != center_word]
                inputs += [tk[center_word]] * len(real_context)
                labels += [tk[w] for w in real_context]
                i += 1
        inputs = np.array(inputs, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32).reshape((len(labels), 1))
        return inputs, labels
