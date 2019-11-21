import os
import numpy as np
import data_to_vector as d2v

data_dir = os.getcwd() + "\\Data"
data = d2v.data2vec(data_dir)

def analyze_data():
    sent_lens = []
    for key in data.keys():
        for sentence in data[key]:
            sent_lens.append(len(sentence))
    ninety_percentile = np.percentile(sent_lens, 90)
    return ninety_percentile

def pad_data():
    final_len = int(analyze_data())
    for key in data.keys():
        padded_data = []
        for sentence in data[key]:
            new_sent = sentence
            if len(sentence) < final_len:
                while len(new_sent) < 15:
                    new_sent.insert(0, "<pad>")
            elif len(sentence) > final_len:
                total_cut = len(sentence) - final_len
                new_sent = new_sent[:-total_cut]
            new_sent.insert(0, "[CLS] ")
            new_sent.append(" [SEP]")
            padded_data.append(new_sent)
        data[key] = padded_data
    return data
