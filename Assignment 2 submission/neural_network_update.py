import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

from preprocess import data_preprocess, feature_extraction_embedding
from tweets_update import Tweets

class Net(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(embed_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # print("x: ", x)

        x = x.float()
        hidden1 = self.layer1(x)
        relu = F.relu(hidden1)
        hidden2 = self.layer2(relu)
        log_probs = F.log_softmax(hidden2, dim=1)
        return log_probs


def predict_val(data, neural_net, dataset):
    new_data = []
    for index in range(len(data)):
        word_embeds = dataset.embeddings(torch.tensor(data[index]))
        sentence_embeds = list(torch.sum(word_embeds, dim=0))
        new_data.append(sentence_embeds)
    output = neural_net(torch.tensor(new_data))
    preds = np.argmax(output.detach().numpy(), axis=1)
    return preds


def predict_test(data, neural_net):
    output = neural_net(torch.from_numpy(data))
    preds = np.argmax(output.detach().numpy(), axis=1)
    return preds


def compare(preds, labels):
    acc = 0
    correct = 0
    total = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            correct+=1
        total+=1
    acc = correct/total
    return acc


def write_testset_prediction(test_data, neural_net, file_name):
    preds = predict_test(test_data, neural_net)

    outfile = open(file_name, 'w')
    outfile.write('ID\tSentiment\n')
    ID = 1
    for pred in preds:
        sentiment_pred = 'pos' if pred==1 else 'neg'
        outfile.write(str(ID)+','+sentiment_pred+'\n')
        ID += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-mv','--max_vocab', help='max vocab size predefined, no limit if set -1', required=False, default=-1)
    parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-fn','--file_name', help='file name', required=False, default='myTest')
    parser.add_argument('-ue','--update_embeds', help='True to update embeds, default is False', action='store_true', default=False)
    parser.add_argument('-lr','--learning_rate', required=False, default=0.005)
    parser.add_argument('-bsz','--batch_size', required=False, default=32)
    args = vars(parser.parse_args())
    print(args)

    dataset = Tweets("./twitter-sentiment.csv", args['clean'], int(args['max_vocab']), args['update_embeds'])
    bsz = int(args['batch_size'])
    train_loader = DataLoader(dataset=dataset,
                             batch_size=bsz,
                             shuffle=True)

    word2idx = dataset.word2idx
    trained_embeddings = dataset.embeddings.weight.detach().numpy()

    embedding_size = trained_embeddings.shape[1]
    hidden_size = 120
    neural_net = Net(embedding_size, hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(neural_net.parameters(), lr=args['learning_rate'], momentum=0.5)

    ### Training loop
    num_epochs = 20
    X_dev = dataset.X_dev
    Y_dev = dataset.Y_dev
    max_val_acc = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (sentence_embed, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = neural_net(sentence_embed)
            target = target.view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.data

        loss_avg = float(total_loss / len(train_loader))
        print("{}/{} loss {:.2f}".format(epoch + 1, num_epochs, loss_avg))

        ## print validation accuracy
        Y_prediction_dev = predict_val(X_dev, neural_net, dataset)
        val_acc = compare(Y_prediction_dev, Y_dev)
        print("Epoch {}, Dev accuracy: {:.4f} %".format(epoch + 1, val_acc))
        if val_acc > max_val_acc:
            max_val_acc = val_acc

    print("Best dev accuracy: {:.4f} %".format(max_val_acc))

    ### Generating test results

    print('\n[Start evaluating on the official test set and dump as {}...]'.format(args['file_name']+'.csv'))
    revs, _ = data_preprocess("./twitter-sentiment-testset.csv", args['clean'], int(args['max_vocab']))
    test_data, _ = feature_extraction_embedding(revs, word2idx, trained_embeddings)
    write_testset_prediction(np.array(test_data), neural_net, args['file_name'] +'.csv')
