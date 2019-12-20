import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pickle

epoch = 10
batch_size = 1
input_size = 100
lr = .0001
langs = ["Chinese", "French", "Greek", "Italian", "Portugese", "European Spanish", "Latin American Spanish"]
lang_count = len(langs)

torch.manual_seed(1)
np.random.seed(1)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4)
        self.rnn = nn.TransformerEncoder(
        	encoder_layer=self.encoder_layer,
            num_layers=6,
        )
        self.f1 = torch.tanh
        self.f2 = nn.Linear(input_size, 64)
        self.f3 = nn.Linear(64, 56)
        self.out = nn.Linear(56, 7)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # h_n shape (n_layers, batch, hidden_size)
        r_out = self.rnn(x)
        h1 = self.f1(r_out)
        h2 = self.f1(self.f2(h1))
        h3 = self.f1(self.f3(h2))
        out = self.out(h3[:, -1, :])
        return out

with open("word_vecs.txt", 'rb') as f:
	data = pickle.load(f)

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
loss_func = F.cross_entropy

d = []
for l in range(lang_count):
    d += [ [line, l] for line in data[langs[l]] ]
np.random.shuffle(d)
train = d[:3*len(d)//5]
val = d[3*len(d)//5:4*len(d)//5]
test = d[4*len(d)//5:]
for ep in range(epoch):
    train_loss_out = 0
    val_loss_out = 0
    y_right = 0
    y_total = 0
    np.random.shuffle(train)
	
    for i in range(0,len(train),batch_size):
        b_x = torch.tensor([b[0] for b in train[i:i+batch_size]])
        b_y = torch.tensor([b[1] for b in train[i:i+batch_size]])

        optimizer.zero_grad()
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        loss.backward()
        optimizer.step()

        train_loss_out += loss.item()
    for i in range(0,len(val),batch_size):
        b_x = torch.tensor([b[0] for b in val[i:i+batch_size]])
        b_y = torch.tensor([b[1] for b in val[i:i+batch_size]])

        test_output = rnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        loss  = loss_func(test_output, b_y)
        #print(pred_y, b_y.tolist())
        y_right += pred_y == b_y.tolist()
        y_total += 1
        val_loss_out += loss.item()
    accuracy = y_right/y_total
    print('Epoch: ', ep, '| train loss:', train_loss_out, '| validation loss:', val_loss_out,'| validation accuracy: %.2f' % accuracy)
for i in range(0,len(val),batch_size):
    b_x = torch.tensor([b[0] for b in test[i:i+batch_size]])
    b_y = torch.tensor([b[1] for b in test[i:i+batch_size]])
	
    test_output = rnn(b_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    #print(pred_y, b_y.tolist())
    y_right += pred_y == b_y.tolist()
    y_total += 1
accuracy = y_right/y_total
print('Test accuracy: %.2f' % accuracy)