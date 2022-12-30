import pickle
import argparse

from numpy import argmax

def read_corpus(english_corpus, foreign_corpus):
    english = [['*'] + e_sent.split() for e_sent in open(english_corpus)]
    foreign = [f_sent.split() for f_sent in open(foreign_corpus)]
    return english, foreign


parser = argparse.ArgumentParser()
parser.add_argument('--iterations',dest='iterations',default=0,
                    help='number of iterations to test')

args = parser.parse_args()
english, spanish = read_corpus('./dev.en', './dev.es')

if int(args.iterations) == 0:
    with open('t_params_ibm1.pickle', 'rb') as handle:
        t_params = pickle.load(handle)

    

    parallel_corpus = zip(english, spanish)

    for sent_idx,(eng_sent,span_sent) in enumerate(parallel_corpus):
        for span_idx,span_word in enumerate(span_sent):
            argmax_dict = dict()
            for eng_idx,eng_word in enumerate(eng_sent):
                pair_key = (eng_word,eng_idx) #span_word)
                t_pair_key = (eng_word,span_word)
                argmax_dict[pair_key] = t_params[t_pair_key]

            argmax_key = max(argmax_dict, key=argmax_dict.get) #get the key with the highest value
            argmax_idx = argmax_key[1]
            eng_idx_in_sent = argmax_idx
            esp_idx_in_sent = span_idx + 1
            print(f'{sent_idx + 1} {eng_idx_in_sent} {esp_idx_in_sent}')
else:
    for i in range(int(args.iterations)):
        with open(f't_params_ibm1_iteration{i+1}.pickle', 'rb') as handle:
            t_params = pickle.load(handle)

        parallel_corpus = zip(english, spanish)
        
        write_lines = []

        for sent_idx,(eng_sent,span_sent) in enumerate(parallel_corpus):
            for span_idx,span_word in enumerate(span_sent):
                argmax_dict = dict()
                for eng_idx,eng_word in enumerate(eng_sent):
                    pair_key = (eng_word,eng_idx) #span_word)
                    t_pair_key = (eng_word,span_word)
                    argmax_dict[pair_key] = t_params[t_pair_key]

                argmax_key = max(argmax_dict, key=argmax_dict.get) #get the key with the highest value
                argmax_idx = argmax_key[1]
                eng_idx_in_sent = argmax_idx
                esp_idx_in_sent = span_idx + 1
                write_lines.append('{0} {1} {2}\n'.format(sent_idx + 1,eng_idx_in_sent,esp_idx_in_sent))

        with open(f'dev_ibm1_iteration{i+1}.out','w') as f:
            f.writelines(write_lines)




