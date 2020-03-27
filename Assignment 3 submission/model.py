
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        #TO DO
        #hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(),
        embed_dim = args.embed_dim
        kernel_num = args.kernel_num
        kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
        dropout = args.dropout
        class_num = args.class_num
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.requires_grad = True

        self.conv1 = nn.Conv2d(1, kernel_num, (kernel_sizes[0], embed_dim))
        self.conv2 = nn.Conv2d(1, kernel_num, (kernel_sizes[1], embed_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (kernel_sizes[2], embed_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)


    def forward(self, x):
        #TO DO
        #input x dim: (batch_size, max_seq_len, D)

        # print("sentence shape: ", x.shape)
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x1 = F.relu(self.conv1(x)).squeeze(3)
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)
        # x1 = F.avg_pool1d(x1, x1.shape[2]).squeeze(2)

        x2 = F.relu(self.conv2(x)).squeeze(3)
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        # x2 = F.avg_pool1d(x2, x2.shape[2]).squeeze(2)

        x3 = F.relu(self.conv3(x)).squeeze(3)
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        # x3 = F.avg_pool1d(x3, x3.shape[2]).squeeze(2)

        x = torch.cat([x1,x2,x3], dim=1)
        # print("x.shape: ", x.shape)

        x = self.dropout(x)
        logit = self.fc(x)
        # print("logit.shape: ", logit.shape)

        return logit
