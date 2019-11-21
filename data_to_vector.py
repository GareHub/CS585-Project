import csv, os
import string

'''
data2vec takes in a string 'dirname', which is a directory name of CSVs.
It returns a dictionary, with each key being a file name within the dictionary,
and the keys being a list of sequences, or lists of individual words.
'''
def data2vec(dirname): 
    directory = os.listdir(dirname)
    class_dictionary = {}
    for filename in directory:
        name = "Data/" + filename
        with open(name, encoding='utf-8') as current_data_set:
            csv_reader = list(csv.reader(current_data_set, delimiter = ','))
            csv_reader = csv_reader[4:]
            class_dictionary[filename] = []
            for x in csv_reader:
                x[2] = x[2].lower()
                x[3] = x[3].lower()
                sequence = x[2:3] + (x[3].strip(string.punctuation).split())
                sequence = sequence[:-1]
                class_dictionary[filename].append(sequence)

    return class_dictionary
