import numpy as np
import pickle
import sys
import argparse

# %matplotlib inline
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',dest='epochs',default=5,
                    help='number of epochs to train')

args = parser.parse_args()

def read_corpus_addnull(english_corpus, foreign_corpus):
    "Reads a corpus and adds in the NULL word."
    english = [['*'] + e_sent.split() for e_sent in open(english_corpus)] #["*"] + 
    foreign = [f_sent.split() for f_sent in open(foreign_corpus)]
    return english, foreign

english_corpus = dict()
spanish_corpus = dict()

en_lines = []
with open('./corpus.en') as f:
    en_lines = f.readlines()

es_lines = []
with open('./corpus.es') as f:
    es_lines = f.readlines()
    
english, spanish = read_corpus_addnull('./corpus.en', './corpus.es')

# check that english sentence has indeed been append with * symbol

n_e = {}

parallel_corpus = zip(english, spanish)

en_lengths = []
es_lengths = []
for e,s in parallel_corpus:
    en_lengths.append(len(e)) # +1 for the NULL english word
    es_lengths.append(len(s))

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

#assign the initial t(f|e)
with open('t_params_ibm1_iteration5.pickle', 'rb') as handle:
    t_params = pickle.load(handle)

#assign initial q(j|i,l,m)
q_params = dict()
for es_len,en_len in zip(es_lengths,en_lengths):
    for i in range(es_len):
        for j in range(en_len):
            key = (j,i,en_len,es_len)
            q_params[key] = 1 / (en_len + 1)

EPOCHS = int(args.epochs)
parallel_corpus = zip(english, spanish)
#EM algorithm
for i in range(EPOCHS):
    counts_e_f = dict()
    counts_e = dict()
    counts_ji_lm = dict()
    counts_ilm = dict()
    delta_dict = dict()
    denominator_dict = dict()

    print("starting iteration " + str((i+1)), file=sys.stderr)
    #set all counts to
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
                for eng_idx2, eng_word2 in enumerate(eng_sent):
                    pair_word2 = (eng_word2, esp_word)
                    q_key = (eng_idx2,esp_idx,len(eng_sent),len(esp_sent))
                    denominator += t_params[pair_word2] * q_params[q_key]
                
                q_key = (eng_idx,esp_idx,len(eng_sent),len(esp_sent))
                delta_dict[pair_idx] = (q_params[q_key] * t_params[pair_word]) / denominator #denominator_dict[(esp_word,esp_idx)] #denominator
                counts_e_f[pair_word] += delta_dict[pair_idx]
                counts_e[eng_word] += delta_dict[pair_idx]
                
                key_jilm = (eng_idx,esp_idx,len(eng_sent),len(esp_sent))
                counts_ji_lm[key_jilm] = counts_ji_lm.setdefault(key_jilm,0) + delta_dict[pair_idx]

                key_ilm = (esp_idx,len(eng_sent),len(esp_sent))
                counts_ilm[key_ilm] = counts_ilm.setdefault(key_ilm,0) + delta_dict[pair_idx]

    for pair in t_params.keys():
        t_params[pair] = counts_e_f[pair] / counts_e[pair[0]] #pair[0] is the english word

    for pair in q_params.keys():
        q_params[pair] = counts_ji_lm[pair] / counts_ilm[(pair[1],pair[2],pair[3])]

    with open(f't_params_ibm2_iteration{i+1}.pickle','wb') as handle:
        pickle.dump(t_params,handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'q_params_ibm2_iteration{i+1}.pickle','wb') as handle:
        pickle.dump(q_params,handle, protocol=pickle.HIGHEST_PROTOCOL)

        

#write t_params to pickle file
with open('t_params_ibm2.pickle','wb') as handle:
    pickle.dump(t_params,handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('q_params.pickle','wb') as handle:
    pickle.dump(q_params,handle, protocol=pickle.HIGHEST_PROTOCOL)



        
            


