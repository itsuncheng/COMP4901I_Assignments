import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from RNN import RNN_LM
from dataloader import get_dataloaders
from tqdm import tqdm
from statistic import loadGloveModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--vocab_size", type=float, default=10000)
    parser.add_argument("--embed_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--win_size", type=int, default=35)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--use_glove", type=bool, default=False)
    args = parser.parse_args()

    # vocab_size = args.vocab_size
    num_epochs = args.num_epochs
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    batch_size = args.batch_size
    win_size = args.win_size
    num_samples = args.num_samples
    early_stop = args.early_stop
    use_glove = args.use_glove

    train_loader, dev_loader, test_loader, vocab_size, vocab = get_dataloaders(batch_size, win_size)

    weight = None
    if use_glove:
        weight = loadGloveModel(vocab, "glove.6B.300d.txt")
    model = RNN_LM(vocab_size, embed_size, hidden_size, num_layers, weight, use_glove)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model, lowest_perplexity = trainer(train_loader, dev_loader, model, optimizer, criterion, num_epochs, early_stop, num_layers, batch_size, hidden_size)
    print("lowest_perplexity: ", lowest_perplexity)

    test(test_loader, model, num_layers, batch_size, hidden_size, criterion)
    generate_words("I.txt", "I", num_layers, hidden_size, vocab, num_samples, model) # starting word to generate from
    generate_words("What.txt", "What", num_layers, hidden_size, vocab, num_samples, model)
    generate_words("Anyway.txt", "anyway", num_layers, hidden_size, vocab, num_samples, model)


def trainer(train_loader,dev_loader, model, optimizer, criterion, num_epochs, early_stop, num_layers, batch_size, hidden_size):
    lowest_perplexity = 1000000;

    for epoch in range(num_epochs):
        loss_log = []
        state = torch.zeros(num_layers, batch_size, hidden_size)

        # for i in range(0, len(train_loader)-1):
        # step = 0
        print('starting to train')
        for inputs, targets in tqdm(train_loader):

            outputs, state = model(inputs, state)
            loss = criterion(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print('starting to validate')
        model.eval()
        valid_loss_log = []
        valid_state = torch.zeros(num_layers, batch_size, hidden_size)

        for valid_inputs, valid_targets in tqdm(dev_loader):
            valid_outputs, valid_state = model(valid_inputs, valid_state)
            valid_loss = criterion(valid_outputs, valid_targets.reshape(-1))
            valid_loss_log.append(valid_loss.item())
        current_perplexity = np.exp(np.mean(valid_loss_log))
        print ('valid loss Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}' .format(epoch+1, num_epochs, np.mean(valid_loss_log), current_perplexity))
        if current_perplexity < lowest_perplexity:
            lowest_perplexity = current_perplexity
        else:
            early_stop -= 1

        if early_stop==0:
            break

    return model, lowest_perplexity


def test(test_loader, model, num_layers, batch_size, hidden_size, criterion):

    test_state = torch.zeros(num_layers, batch_size, hidden_size)
    test_loss_log = []
    # test set Perplexity
    for test_inputs, test_targets in tqdm(test_loader):
        test_outputs, test_state = model(test_inputs, test_state)
        test_loss = criterion(test_outputs, test_targets.reshape(-1))
        test_loss_log.append(test_loss.item())
    perplexity = np.exp(np.mean(test_loss_log))
    print ('Test Set Perplexity: {:5.2f}' .format(perplexity))


def generate_words(filename, first_word, num_layers, hidden_size, vocab, num_samples, model):

    with torch.no_grad():
        f = open(filename, 'w')
        state = torch.zeros(num_layers, 1, hidden_size)

        input = torch.LongTensor([[vocab.word2index[first_word]]])
        print("first input", input)
        f.write(first_word + ' ')
        for i in range(num_samples):
            # Forward propagate RNN
            output, state = model(input, state)
            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            word = vocab.index2word[word_id]

            if word == '<eos>':
                word = '\n'
            elif word == '<sos>':
                word = ''
            else:
                word = word + ' '

            f.write(word)
            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))




if __name__=="__main__":
    main()
