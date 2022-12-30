import timeit
import numpy as np
import pickle
import sys
import argparse

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
# %matplotlib inline

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',dest='epochs',default=5,
                    help='number of epochs to train')

args = parser.parse_args()

def read_corpus_addnull(english_corpus, foreign_corpus):
    "Reads a corpus and adds in the NULL word."
    english = [["*"] + e_sent.split() for e_sent in open(english_corpus)]
    foreign = [f_sent.split() for f_sent in open(foreign_corpus)]
    return english, foreign

english_corpus = dict()
spanish_corpus = dict()

en_lines = []
#with open('./corpus.en') as f:
with open('./corpus.en') as f:
    en_lines = f.readlines()

es_lines = []
with open('./corpus.es') as f:
    es_lines = f.readlines()


en_lengths = []
es_lengths = []
for i in range(len(en_lines)):
    en_lengths.append(len(en_lines[i].split()))
    es_lengths.append(len(es_lines[i].split()))
    
english, spanish = read_corpus_addnull('./corpus.en', './corpus.es')

# check that english sentence has indeed been append with * symbol

n_e = {}
parallel_corpus = zip(english, spanish)
wordpairs = set()
for e, s in parallel_corpus:
    for e_j in e:
        for s_i in s:
            wordpair = (e_j, s_i)
            if wordpair not in wordpairs:
                wordpairs.add(wordpair)
                if not e_j in n_e:
                    n_e[e_j] = 0
                n_e[e_j]+=1

# Sanity check n_e for NULL should be equal to the  number of unique spanish words.
# This is because the special English word NULL can be aligned to any foreign word
#in the corpus.

spanish_words = set()
for sent in spanish:
    for word in sent:
        spanish_words.add(word)
# print(len(spanish_words), n_e["*"] ==len(spanish_words))

import operator

t_params = dict()
#assign the initial t(f|e)

for (eng_word,esp_word) in wordpairs:
    pair = (eng_word,esp_word)
    t_params[pair] = 1 / n_e[eng_word]
    

counts_ji_lm = dict()
counts_ilm = dict()

EPOCHS = int(args.epochs)
parallel_corpus = zip(english, spanish)
#EM algorithm
for i in range(EPOCHS):
    counts_e_f = dict()
    counts_e = dict()
    delta_dict = dict()
    denominator_dict = dict()

    for (eng_word,esp_word) in wordpairs:
        pair = (eng_word,esp_word)
        counts_e_f[pair] = 0
        if eng_word not in counts_e.keys():
            counts_e[eng_word] = 0

    parallel_corpus = zip(english, spanish)
    for sent_idx,(eng_sent,esp_sent) in enumerate(parallel_corpus):
        for esp_idx,esp_word in enumerate(esp_sent):
            for eng_idx,eng_word in enumerate(eng_sent):
                pair_word = (eng_word,esp_word)
                pair_idx = (eng_idx,esp_idx)

                denominator = 0
                for eng_idx2, eng_word2 in enumerate(eng_sent): #.copy()):
                    pair_word2 = (eng_word2, esp_word) #(e,f)
                    denominator += t_params[pair_word2]

                delta_dict[pair_idx] = t_params[pair_word] / denominator
                counts_e_f[pair_word] += delta_dict[pair_idx]
                counts_e[eng_word] += delta_dict[pair_idx]

    for pair in t_params.keys():
        t_params[pair] = counts_e_f[pair] / counts_e[pair[0]] #pair[0] is the english word

    with open(f't_params_ibm1_iteration{i+1}.pickle','wb') as handle:
        pickle.dump(t_params,handle, protocol=pickle.HIGHEST_PROTOCOL)

        

#write t_params to pickle file
with open('t_params_ibm1.pickle','wb') as handle:
    pickle.dump(t_params,handle, protocol=pickle.HIGHEST_PROTOCOL)


        
            


