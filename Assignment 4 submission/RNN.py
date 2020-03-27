import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_LM(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, weight, use_glove):
        super(RNN_LM, self).__init__()

        if use_glove:
            self.emb = nn.Embedding.from_pretrained(weight)
            self.emb.weight.requires_grad = True
        else:
            self.emb = nn.Embedding(vocab_size, emb_size)
        self.RNN = nn.RNN(emb_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, h):
        inputs = self.emb(inputs)
        # print("after emb: ", inputs.shape)
        # print("hidden size: ", h.shape)
        out, h = self.RNN(inputs, h)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)

        return out, h
