import torch
import torch.utils.data as data
from statistic import Vocab
from statistic import Lang
from preprocess import preprocess

UNK_INDEX = 0

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, target, vocab, batch_size):
        self.content = data[:int(len(data)/batch_size) * batch_size]    # remove data in the end that does not fit with batch size
        self.target = target[:int(len(target)/batch_size) * batch_size]
        self.vocab = vocab
        self.num_total_win = len(self.content)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        cont = self.content[index]
        cont = self.tokenize(cont)
        targ = self.target[index]
        targ = self.tokenize(targ)
        return torch.LongTensor(cont), torch.LongTensor(targ)

    def __len__(self):
        return self.num_total_win
    def tokenize(self, sentence):
        return [self.vocab.word2index[word] if word in self.vocab.word2index else UNK_INDEX for word in sentence]


def get_dataloaders(batch_size, win_size):
    vocab = Vocab()
    vocab, statistic = Lang(vocab, "train.txt")

    train_data_X, train_data_Y = preprocess("train.txt", win_size)
    dev_data_X, dev_data_Y = preprocess("valid.txt", win_size)
    test_data_X, test_data_Y = preprocess("test.txt",win_size)
    train = Dataset(train_data_X, train_data_Y, vocab, batch_size)
    dev = Dataset(dev_data_X, dev_data_Y, vocab, batch_size)
    test = Dataset(test_data_X, test_data_Y, vocab, batch_size)
    print(statistic)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev,
                                                    batch_size=batch_size,
                                                    shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test,
                                                    batch_size=batch_size,
                                                    shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, statistic["vocab_size"], vocab
