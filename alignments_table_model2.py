import pickle
import pandas as pd
import matplotlib as mpl
import argparse
from tabulate import tabulate


def read_corpus(english_corpus, foreign_corpus):
    english = [['*'] + e_sent.split() for e_sent in open(english_corpus)]
    foreign = [f_sent.split() for f_sent in open(foreign_corpus)]
    return english, foreign

english, spanish = read_corpus('./dev.en', './dev.es')


with open('dev_ibm2_iteration5.out','r') as f:
    lines = f.readlines()
    lines = [line.rstrip().split() for line in lines]
    lines = [line for line in lines if line[0] == '1']


one_hot_encodings = dict()
for line in lines:
    one_hot_enc = []

    for i in range(16):
        if i == int(line[1]):
            one_hot_enc.append(1)
        else:
            one_hot_enc.append(0)
    
    one_hot_encodings[int(line[2])] = one_hot_enc

one_hot_encs = []

for i in range(20):
    if i+1 in one_hot_encodings.keys():
        one_hot_encs.append(one_hot_encodings[i+1])
    else:
        one_hot_encs.append([0] * 16)




df = pd.DataFrame(one_hot_encs,
                  index=pd.Index(spanish[0]),
                  columns=pd.Index(english[0]))
#df.style
print(tabulate(df,headers = 'keys',tablefmt='psql'))




