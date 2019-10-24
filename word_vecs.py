
import pandas as pd
import csv

glove_data_file = "glove.6B\glove.6B.100d.txt" #Change to path of target glove file

words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

#Returns a word vector for a given word using glove
def vec(w):
    return words.loc[w.lower()].as_matrix()