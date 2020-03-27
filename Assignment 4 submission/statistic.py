import pandas as pd
from collections import Counter
import numpy as np
from preprocess import clean
from tqdm import tqdm
import torch

UNK_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2

class Vocab():
    def __init__(self):
        self.word2index = {"<unk>":UNK_INDEX, "<sos>": SOS_INDEX, "<eos>": EOS_INDEX }
        self.word2count = {"<unk>": 0}
        self.index2word = {UNK_INDEX: "<unk>", SOS_INDEX: "<sos>", EOS_INDEX: "<eos>" }
        self.n_words = 3 # Count default tokens
        self.word_num = 0
    def index_words(self, sentence):
        for word in sentence:
            self.word_num+=1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1


def Lang(vocab, filename):
    statistic = {"sent_num":0, "word_num":0, "vocab_size":0, "UNK_num": 0, "top_ten_words":[] }
    ############################################################
    # TO DO
    #build vocabulary and statistic

    sent_num = 0

    f = open(filename, "r+")
    for line in f:
        line = str(line).rstrip()
        sent = line.split()  # add clean for training but remove clean for calculating statistics 
        vocab.index_words(sent)
        sent_num += 1

    statistic["sent_num"] = sent_num
    statistic["word_num"] = vocab.word_num
    statistic["vocab_size"] = vocab.n_words
    statistic["UNK_num"] = float("{0:.5f}".format(vocab.word2count["<unk>"]/vocab.word_num))
    statistic["top_ten_words"] = [word for word in dict(Counter(vocab.word2count).most_common(10))]

    ############################################################
    return vocab, statistic


def loadGloveModel(vocab, gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')

    words = []
    embeddings = []

    for line in tqdm(f):
        line = line.rstrip().split()
        word = line[0]
        embedding = [float(val) for val in line[1:]]
        words.append(word)
        embeddings.append(embedding)

    weight = []
    for word in vocab.word2index:
        if word in words:
            index = words.index(word)
            weight.append(embeddings[index])
        else:
            weight.append([0] * 300)

    weight = torch.FloatTensor(weight)

    return weight
