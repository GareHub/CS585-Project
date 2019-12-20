import CSV_to_Dict as dv
import pandas as pd
import csv
import pickle
import numpy as np

indict = dv.import_data('Data')

# GloVe files were not included on GitHub due to extremely large sizes (up to 1GB)
GLOVE_FILE = "GloVe\glove.6B.100d.txt" #Change to path of target glove file

words = pd.read_table(GLOVE_FILE, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

# Find number and length of GloVe embeddings
with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        pass
n_vec = i + 1
hidden_dim = len(line.split(' ')) - 1

# Find average vector to represent unknown tokens
vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)
with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)
average_vec = np.mean(vecs, axis=0)

#Returns a word vector for a given word from GloVe dictionary
def vec(w):
	try:
		x = words.loc[w.lower()].values()
	except:
		x = average_vec
	return x

# Create dict for languages and dump to text file. 
# In word_vecs_2.0.txt, word vector lists are paired with their original sentences to enable error analysis on sentences
# This takes a while so print statements show progress
print("Starting...")
outdict = {}
outdict["Chinese"] = [([vec(w) for w in sen],sen) for sen in indict['clc_chineseL1.csv']]
print("Chinese done")
outdict["French"] = [([vec(w) for w in sen],sen) for sen in indict['clc_frenchL1.csv']]
print("French done")
outdict["Greek"] = [([vec(w) for w in sen],sen) for sen in indict['clc_greekL1.csv']]
print("Greek done")
outdict["Italian"] = [([vec(w) for w in sen],sen) for sen in indict['clc_italianL1.csv']]
print("Italian done")
outdict["Portugese"] = [([vec(w) for w in sen],sen) for sen in indict['clc_portugeseL1.csv']]
print("Portugese done")
outdict["European Spanish"] = [([vec(w) for w in sen],sen) for sen in indict['clc_spanishL1_eu.csv']]
print("EU Spanish done")
outdict["Latin American Spanish"] = [([vec(w) for w in sen],sen) for sen in indict['clc_spanishL1_la.csv']]
print("LA Spanish done")
with open("word_vecs_2.0.txt", 'wb') as outf:
	pickle.dump(outdict, outf)
