import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pickle

# Defining parameters
EPOCH = 10
BATCH_SIZE = 1
INPUT_SIZE = 100
LR = .0001
langs = ["Chinese", "French", "Greek", "Italian", "Portugese", "European Spanish", "Latin American Spanish"]
lang_count = len(langs)

torch.manual_seed(1)
np.random.seed(1)

# Defining our model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
        )
        self.f1 = torch.tanh
        self.f2 = nn.Linear(100, 64)
        self.f3 = nn.Linear(64, 56)
        self.out = nn.Linear(56, 7)

    def forward(self, x):
        r_out, h_n = self.rnn(x, None)
        h1 = self.f1(r_out)
        h2 = self.f1(self.f2(h1))
        h3 = self.f1(self.f3(h2))
        out = self.out(h3[:, -1, :])
        return out

# Open file containg word vectors
with open("word_vecs.txt", 'rb') as f:
	data = pickle.load(f)

# Initialize model
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = F.cross_entropy

# Convert language dict to train, validation, and test splits
d = []
for l in range(lang_count):
    d += [ [line, l] for line in data[langs[l]] ]
np.random.shuffle(d)
train = d[:3*len(d)//5]
val = d[3*len(d)//5:4*len(d)//5]
test = d[4*len(d)//5:]

# Run epochs
for epoch in range(EPOCH):
    train_loss_out = 0
    val_loss_out = 0
    y_right = 0
    y_total = 0
    np.random.shuffle(train)
	# Training
    for i in range(0,len(train),BATCH_SIZE):
        b_x = torch.tensor([b[0] for b in train[i:i+BATCH_SIZE]])
        b_y = torch.tensor([b[1] for b in train[i:i+BATCH_SIZE]])

        optimizer.zero_grad()
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        loss.backward()
        optimizer.step()

        train_loss_out += loss.item()
    # Validation
    for i in range(0,len(val),BATCH_SIZE):
        b_x = torch.tensor([b[0] for b in val[i:i+BATCH_SIZE]])
        b_y = torch.tensor([b[1] for b in val[i:i+BATCH_SIZE]])

        test_output = rnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        loss  = loss_func(test_output, b_y)
        #print(pred_y, b_y.tolist())
        y_right += pred_y == b_y.tolist()
        y_total += 1
        val_loss_out += loss.item()
    accuracy = y_right/y_total
    print('Epoch: ', epoch, '| train loss:', train_loss_out, '| validation loss:', val_loss_out,'| validation accuracy: %.2f' % accuracy)
# Testing
for i in range(0,len(val),BATCH_SIZE):
    b_x = torch.tensor([b[0] for b in test[i:i+BATCH_SIZE]])
    b_y = torch.tensor([b[1] for b in test[i:i+BATCH_SIZE]])
	
    test_output = rnn(b_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    y_right += pred_y == b_y.tolist()
    y_total += 1

# Print accuracy
accuracy = y_right/y_total
print('Test accuracy: %.2f' % accuracy)