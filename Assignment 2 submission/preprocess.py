import numpy as np
import re
import collections
import io

def read_data(_file, cleaning):
    revs = []
    max_len = 0
    words_list = []
    with io.open(_file, "r",  encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            ID, label, sentence = line.split('\t')
            label_idx = 1 if label=='pos' else 0 # 1 for pos and 0 for neg
            rev = []
            rev.append(sentence.strip())

            if cleaning:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            revs.append({'y':label_idx, 'txt':orig_rev})
            words_list += orig_rev.split()
    return revs, words_list


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_info(revs, words_list):
    """
    TODO:
        First check what is revs. Then calculate max len among the sentences and the number of the total words
        in the data.
        nb_sent, max_len, word_count are scalars
    """
    nb_sent, max_len, word_count = 0, 0, 0
    nb_sent = len(revs) # assume each review is a sentence
    for rev in revs:
        len_rev = len(rev["txt"])
        if len_rev > max_len:
            max_len = len_rev
        word_count += len(rev["txt"].split())
    return nb_sent, max_len, word_count


def data_preprocess(_file, cleaning, max_vocab_size):
    revs, words_list = read_data(_file, cleaning)
    nb_sent, max_len, word_count = get_info(revs, words_list)

    ## get vocab word2idx, top_10_words
    word2idx = np.load('word2index.npy').item()
    if max_vocab_size == -1:
        max_vocab_size = len(word2idx)
    word2idx = dict(list(word2idx.items())[:max_vocab_size])

    top_10_words = list(word2idx.keys())[:10]
    # data analysis
    print("Number of words: ", word_count)
    print("Max sentence length: ", max_len)
    print("Number of sentences: ", nb_sent)
    print("Number of vocabulary: ", len(word2idx))
    print("Top 10 most frequently words", top_10_words)

    return revs, word2idx

def feature_extraction_embedding(revs, word2idx, emb_mat):
    data = []
    label = []
    # counter = 0
    for sent_info in revs:
        words = sent_info["txt"].strip().split()
        sum_vector = np.zeros(emb_mat.shape[1])
        for word in words:
            if word not in word2idx:
                word = 'UNK'
            word_vector = emb_mat[word2idx[word]]
            sum_vector += word_vector

        data.append(list(sum_vector))
        label.append([sent_info['y']])

    return np.array(data), np.array(label)


def normalization(data):
    """
    TODO:
        Normalize each dimension of the data into mean=0 and std=1
    """

    np.seterr(divide='ignore', invalid='ignore') ## might encounter division over zero, so let numpy ignore this error
    print("data before normalization: ", data)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    print("data after normalization: ", data)

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f','--file', help='input csv file', required=False, default='./twitter-sentiment.csv')
    parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv','--max_vocab', help='max vocab size predefined, no limit if set -1', required=False, default=-1)
    args = vars(parser.parse_args())
    print(args)

    revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))

    word_emb_mat = np.loadtxt('w_emb_mat.txt')
    data, label = feature_extraction_embedding(revs, word2idx, word_emb_mat)
    data = normalization(data)
