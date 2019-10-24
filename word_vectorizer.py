import data_to_vector as dv
import pandas as pd
import csv
import pickle
import numpy as np

indict = dv.data2vec('Data')

GLOVE_FILE = "WordEmbedding\glove.6B\glove.6B.100d.txt" #Change to path of target glove file

words = pd.read_table(GLOVE_FILE, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        pass
n_vec = i + 1
hidden_dim = len(line.split(' ')) - 1

vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

average_vec = np.mean(vecs, axis=0)

#Returns a word vector for a given word using glove
def vec(w):
	try:
		x = words.loc[w.lower()].values()
	except:
		x = average_vec
	return x

outdict = {}
outdict["Chinese"] = [[vec(w) for w in sen] for sen in indict['clc_chineseL1.csv']]
outdict["French"] = [[vec(w) for w in sen] for sen in indict['clc_frenchL1.csv']]
outdict["Greek"] = [[vec(w) for w in sen] for sen in indict['clc_greekL1.csv']]
outdict["Italian"] = [[vec(w) for w in sen] for sen in indict['clc_italianL1.csv']]
outdict["Portugese"] = [[vec(w) for w in sen] for sen in indict['clc_portugeseL1.csv']]
outdict["European Spanish"] = [[vec(w) for w in sen] for sen in indict['clc_spanishL1_eu.csv']]
outdict["Latin American Spanish"] = [[vec(w) for w in sen] for sen in indict['clc_spanishL1_la.csv']]
with open("word_vecs.txt", 'wb') as outf:
	pickle.dump(outdict, outf)
