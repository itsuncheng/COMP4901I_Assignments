import torch
import torch.nn as nn
from torch.utils.data import Dataset
from preprocess import data_preprocess
import numpy as np
from sklearn.model_selection import train_test_split

def feature_extraction_index(revs, word2idx):
    data = []
    label = []
    for sent_info in revs:
        words = sent_info["txt"].strip().split()
        sentence_index = []
        for word in words:
            if word not in word2idx:
                word = 'UNK'
            word_index = word2idx[word]
            sentence_index.append(word_index)

        if len(sentence_index) == 0:
            continue

        data.append(sentence_index)
        label.append([sent_info['y']])

    return data, label


class Tweets(Dataset):
    def __init__(self, filename, cleaning, max_vocab_size, update_embeds):
        revs, word2idx = data_preprocess(filename, cleaning, max_vocab_size)
        data, label = feature_extraction_index(revs, word2idx)
        word_emb_mat = np.loadtxt('w_emb_mat.txt')

        # data = normalization(data)
        X_train, X_dev, Y_train, Y_dev = train_test_split(data, label, test_size=0.2, random_state=0)
        # print("X_train.shape: ", X_train.shape)
        self.data = X_train
        self.label = Y_train
        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.word2idx = word2idx
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_emb_mat), freeze = not update_embeds)

    def __getitem__(self, index):
        word_embeds = self.embeddings(torch.tensor(self.data[index]))
        sentence_embeds = torch.sum(word_embeds, dim=0)
        target = torch.tensor(self.label[index])
        # print("sentence_embeds: ", sentence_embeds)
        # print("sentence_embeds.shape: ", sentence_embeds.shape)
        # print("target.shape: ", target.shape)

        return sentence_embeds, target

    def __len__(self):
        return len(self.data)
