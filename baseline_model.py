import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pickle

EPOCH = 10
BATCH_SIZE = 1
INPUT_SIZE = 100
LR = .0003
langs = ["Chinese", "French", "Greek", "Italian", "Portugese", "European Spanish", "Latin American Spanish"]
lang_count = len(langs)

torch.manual_seed(1)
np.random.seed(1)

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
        self.out = nn.Linear(100, 7)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # h_n shape (n_layers, batch, hidden_size)
        r_out, h_n = self.rnn(x, None)
        h1 = self.f1(r_out)
        out = self.out(h1[:, -1, :])
        return out

with open("word_vecs.txt", 'rb') as f:
	data = pickle.load(f)

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = F.cross_entropy

d = []
for l in range(lang_count):
	d += [ [line, l] for line in data[langs[l]] ]
np.random.shuffle(d)
train = d[:3*len(d)//5]
val = d[3*len(d)//5:4*len(d)//5]
test = d[4*len(d)//5:]
for epoch in range(EPOCH):
	loss_out = 0
	y_right = 0
	y_total = 0
	np.random.shuffle(train)
	
	for i in range(0,len(train),BATCH_SIZE):
		b_x = torch.tensor([b[0] for b in train[i:i+BATCH_SIZE]])
		b_y = torch.tensor([b[1] for b in train[i:i+BATCH_SIZE]])

		optimizer.zero_grad()
		output = rnn(b_x)

		loss = loss_func(output, b_y)
		loss.backward()
		optimizer.step()

		loss_out += loss.item()
	for i in range(0,len(val),BATCH_SIZE):
		b_x = torch.tensor([b[0] for b in val[i:i+BATCH_SIZE]])
		b_y = torch.tensor([b[1] for b in val[i:i+BATCH_SIZE]])
		
		test_output = rnn(b_x)
		pred_y = torch.max(test_output, 1)[1].data.numpy()
		#print(pred_y, b_y.tolist())
		y_right += pred_y == b_y.tolist()
		y_total += 1
	accuracy = y_right/y_total
	print('Epoch: ', epoch, '| train loss:', loss_out, '| validation accuracy: %.2f' % accuracy)
for i in range(0,len(val),BATCH_SIZE):
	b_x = torch.tensor([b[0] for b in test[i:i+BATCH_SIZE]])
	b_y = torch.tensor([b[1] for b in test[i:i+BATCH_SIZE]])
	
	test_output = rnn(b_x)
	pred_y = torch.max(test_output, 1)[1].data.numpy()
	#print(pred_y, b_y.tolist())
	y_right += pred_y == b_y.tolist()
	y_total += 1
accuracy = y_right/y_total
print('Test accuracy: %.2f' % accuracy)
