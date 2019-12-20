import csv, os
import string

'''
import_data takes in a string 'dirname', which is a directory name of CSVs.
It returns a dictionary, with each key being a file name within the dictionary,
and the keys being a list of sequences, or lists of individual words.
'''
def import_data(dirname): 
    directory = os.listdir(dirname)
    class_dictionary = {}
    for filename in directory:
        name = "Data/" + filename
        with open(name, 'r', encoding='utf-8') as current_data_set:
            csv_reader = csv.reader(current_data_set)
            lst_csv = list(csv_reader)[4:]
            class_dictionary[filename] = []
            for x in lst_csv:
                x[2] = x[2].lower()
                x[3] = x[3].lower()
                sequence = x[2:3] + (x[3].translate(str.maketrans('.,;','   ')).split())
                sequence = sequence[:-1]
                class_dictionary[filename].append(sequence)

    return class_dictionary
