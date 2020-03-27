import pandas as pd
import re
import numpy as np
import pickle
from collections import Counter

import torch
import torch.utils.data as data

PAD_INDEX = 0
UNK_INDEX = 1
def clean(sent):
    # clean the data
    ############################################################
    # TO DO
    ############################################################
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " \( ", sent)
    sent = re.sub(r"\)", " \) ", sent)
    sent = re.sub(r"\?", " \? ", sent)
    sent = sent.strip().lower()
    return sent

class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX ,"UNK":UNK_INDEX }
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK" }
        self.n_words = 2 # Count default tokens
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

def Lang(vocab, file_name):
    statistic = {"sent_num":0, "word_num":0, "vocab_size":0, "top_ten_words":[], "max_len":0, "avg_len":0, "len_std":0, "class_distribution":{} }
    df = pd.read_csv(file_name)
    statistic["sent_num"] = len(df)
    sent_len_list = []
    ############################################################
    # TO DO
    #build vocabulary and statistic
    sent_list = list(df["content"])
    for sent in sent_list:
        sent = clean(str(sent).strip()).split()
        vocab.index_words(sent)
        sent_len_list.append(len(sent))

    rating_list = list(df["rating"])
    class_dist_dict = dict(Counter(rating_list))

    statistic["word_num"] = vocab.word_num
    statistic["vocab_size"] = vocab.n_words
    statistic["top_ten_words"] = [word for word in dict(Counter(vocab.word2count).most_common(10))]
    statistic["max_len"] = max(sent_len_list)
    statistic["avg_len"] = sum(sent_len_list) / len(sent_len_list)
    statistic["len_std"] = np.std(sent_len_list)
    statistic["class_distribution"] = class_dist_dict


    ############################################################
    return vocab, statistic


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        self.id, self.X, self.y = data
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ind = self.id[index]
        X = self.tokenize(self.X[index])
        if(self.y is not None):
            y = self.y[index]
            return X, y, ind
        else:
            return X, ind

    def __len__(self):
        return self.num_total_seqs
    def tokenize(self, sentence):
        return [self.vocab.word2index[word] if word in self.vocab.word2index else UNK_INDEX for word in sentence]

def preprocess(filename, max_len=200, test=False):
    df = pd.read_csv(filename)
    id_ = [] # review id
    rating = [] # rating
    content = [] #review content

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            rating.append(int(df['rating'][i]))
        sentence = clean(str(df['content'][i]).strip())
        sentence = sentence.split()
        sent_len = len(sentence)
        # here we pad the sequence for whole training set, you can also try to do dynamic padding for each batch by customize collate_fn function
        # if you do dynamic padding and report it, we will give 1 points bonus
        # if sent_len>max_len:
        #     content.append(sentence[:max_len])
        # else:
        #     content.append(sentence+["PAD"]*(max_len-sent_len))
        content.append(sentence)

    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) ==len(rating)
        return (id_, content, rating)


def my_collate(batch, test=False):
    # print("batch: ", batch)
    id_ = [item[2] for item in batch]
    content = [item[0] for item in batch]
    new_content = []
    #dynamic padding: pad each sentence to max_len of batch
    # print("id_: ", id)
    # print("content: ", content)
    max_len = 0
    for sent in content:
        if len(sent)  > max_len:
            max_len = len(sent)

    for sent in content:
        sent_len = len(sent)
        if sent_len>max_len:
            new_content.append(sent[:max_len])
        else:
            padded_sent = sent+[PAD_INDEX]*(max_len-sent_len)
            new_content.append(padded_sent)

    if not test:
        rating = [item[1] for item in batch]
        # print("rating: ", rating)
        return (torch.LongTensor(new_content), torch.LongTensor(rating), torch.LongTensor(id_))
    else:
        return (torch.LongTensor(new_content), torch.LongTensor(id_))




def get_dataloaders(batch_size, max_len):
    vocab = Vocab()
    vocab, statistic = Lang(vocab, "train.csv")

    train_data = preprocess("train.csv", max_len)
    dev_data = preprocess("dev.csv", max_len)
    test_data = preprocess("test.csv",max_len, test=True)
    train = Dataset(train_data, vocab)
    dev = Dataset(dev_data, vocab)
    test = Dataset(test_data, vocab)
    print(statistic)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    collate_fn=my_collate)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    collate_fn=my_collate)
    data_loader_test = torch.utils.data.DataLoader(dataset=test,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    collate_fn=my_collate)
    return data_loader_tr, data_loader_dev, data_loader_test, statistic["vocab_size"]
