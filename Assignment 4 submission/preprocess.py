import re

def clean(sent):
    # clean the data
    ############################################################
    # TO DO
    ############################################################
    sent = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = sent.strip()
    return sent


def preprocess(filename, win_size):

    word_list = []
    f = open(filename, "r+")
    line_count = 0
    for line in f:
        if line_count == 1000:
            break
        line_count +=1
        line = line.rstrip()
        if len(line) == 0:
            line = '\n'
        sent = clean(line).split()
        sent = ["<sos>"] + sent + ["<eos>"]
        for word in sent:
            word_list.append(word)

    win_word_list = []
    target_word_list = []

    for i in range(int(len(word_list)/win_size)):
        win_word_list.append(word_list[i*win_size: (i+1)*win_size])
        target_word_list.append(word_list[i*win_size + 1: (i+1)*win_size + 1])

    return win_word_list, target_word_list
