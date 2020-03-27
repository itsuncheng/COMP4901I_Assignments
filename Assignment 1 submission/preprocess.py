import numpy as np
import re
import collections
import io

def read_data(_file, cleaning, n):
    # Added n as argument for n-gram
    revs = []
    max_len = 0
    words_list = []
    counter=0
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
            list = orig_rev.split()
            words_list += [tuple(list[i:i+n]) for i in range(len(list)-n+1)]

            counter+=1
            if counter%200 ==0:
                print("Reading line at: ", counter)

    return revs, words_list


def load_dict():
    infile = open("dictionary.txt","r+")
    dictionary = []
    for word in infile:
        word = word.rstrip()
        dictionary.append(word)
    return dictionary


def load_stopwords():
    infile = open("stop_words.txt","r+")
    stopwords = []
    for word in infile:
        word = word.rstrip()
        stopwords.append(word)
    return stopwords


def clean_str(string):
    """
    TODO: Data cleaning
    """

    string = string.strip().lower()
    puncs = """!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"""

    dictionary = load_dict()
    stopwords = load_stopwords()
    word_list = re.findall(r"\w+|[^\w\s]", string, re.UNICODE)
    string_list = []
    for word in word_list:
        if word in stopwords:
            string_list.append("") # stopwords
        elif word in puncs:
            string_list.append("PUNC") # punctuation
        elif bool(re.match("\d+", word)):
            string_list.append("N") # number
        elif bool(re.match("[a-z\d+]+(.com)", word)):
            string_list.append("HTML") # html link
        elif word in dictionary:
            string_list.append(word)
        else:
            string_list.append("UNK") # unknown words
    string = " ".join(string_list)

    return string


def build_vocab(words_list, max_vocab_size=-1):
    """
    TODO:
        Build a word dictionary, use max_vocab_size to limit the total number of vocabulary.
        if max_vocab_size==-1, then use all possible words as vocabulary. The dictionary should look like:
        ex:
            word2idx = { 'UNK': 0, 'i': 1, 'love': 2, 'nlp': 3, ... }

        top_10_words is a list of the top 10 frequently words appeared
        ex:
            top_10_words = ['a','b','c','d','e','f','g','h','i','j']
    """

    c = collections.Counter(words_list)
    if max_vocab_size == -1:
        max_vocab_size = len(dict(c))
    freq_dict = dict(c.most_common(max_vocab_size-1)) # UNK excluded
    vocab_list =  [word for word in freq_dict]
    word2idx = {v:k for k,v in enumerate(vocab_list, 1)}
    word2idx['UNK'] = 0

    top_10_words = [word for word in dict(c.most_common(10))]

    return word2idx, top_10_words


def get_info(revs, words_list, n):
    """
    TODO:
        First check what is revs. Then calculate max len among the sentences and the number of the total words
        in the data.
        nb_sent, max_len, word_count are scalars
    """
    # Added n as argument for n-gram
    nb_sent, max_len, word_count = 0, 0, 0
    nb_sent = len(revs) # assume each review is a sentence
    for rev in revs:
        len_rev = len(rev["txt"])
        if len_rev > max_len:
            max_len = len_rev
        list = rev["txt"].split()
        list += [tuple(list[i:i+n]) for i in range(len(list)-n+1)]
        word_count += len(list)
    return nb_sent, max_len, word_count


def data_preprocess(_file, cleaning, max_vocab_size, n):
    revs, words_list = read_data(_file, cleaning, n)
    nb_sent, max_len, word_count = get_info(revs, words_list, n)
    word2idx, top_10_words = build_vocab(words_list, max_vocab_size)
    # data analysis
    print("Number of words: ", word_count)
    print("Max sentence length: ", max_len)
    print("Number of sentences: ", nb_sent)
    print("Number of vocabulary: ", len(word2idx))
    print("Top 10 most frequently words", top_10_words)

    return revs, word2idx


def feature_extraction_bow(revs, word2idx, n):
    """
    TODO:
        Convert sentences into vectors using BoW.
        data is a 2-D array with the size (nb_sentence*nb_vocab)
        label is a 2-D array with the size (nb_sentence*1)
    """
    # Added n as argument for n-gram
    data = []
    label = []
    counter = 0
    for sent_info in revs:
        sent_data = []
        words = sent_info["txt"].strip().split()
        sent_words = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        sent_words_dict = dict(collections.Counter(sent_words))

        for vocab_word in word2idx:
            if vocab_word in sent_words_dict:
                sent_data.append(sent_words_dict[vocab_word])
            else:
                sent_data.append(0)
        data.append(sent_data)

        label.append([sent_info['y']])
        # counter+=1
        # if counter % 1000==0:
        #     print(counter)
    return np.array(data), np.array(label)


def normalization(data):
    """
    TODO:
        Normalize each dimension of the data into mean=0 and std=1
    """

    np.seterr(divide='ignore', invalid='ignore') ## might encounter division over zero, so let numpy ignore this error
    print("data before normalization: ", data)
    data = (data - data.mean()) / data.std()
    print("data after normalization: ", data)

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f','--file', help='input csv file', required=False, default='./twitter-sentiment.csv')
    parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv','--max_vocab', help='max vocab size predefined, no limit if set -1', required=False, default=-1)
    parser.add_argument('-ng','--ngram', help='value of ngram', required=False, default=1)
    args = vars(parser.parse_args())
    print(args)

    revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']), int(args['ngram']))

    data, label = feature_extraction_bow(revs, word2idx, int(args['ngram']))
    data = normalization(data)
