import random
import re
import numpy as np

def get_loanwords(Xw):
    Xnew =[[],[]]
    for i,s in enumerate(Xw[0]):
        index_array = list(range(len(s)))
        np.random.shuffle(index_array)
        Xnew[0].append(p.sub(' ',' _ , '.join([s[idx] for idx in index_array]).strip()))
        s2 = Xw[1][i]
        Xnew[1].append(p.sub(' ',' _ , '.join([s2[idx] for idx in index_array]).strip()))
    return Xnew

p = re.compile('\s+')
q = re.compile('\_\s*\_')

#sbol = ['_','^','`']

Xw = [] #loan_words
for i,dS in enumerate(['.en','.ko']):
    with open('en_es_data/train4_before'+dS,'r') as f:
        Xw.append(f.read().split('\n')[:1200000])

Xw = [[s.split(',') for s in Xw[i]] for i in range(2)]

nS = 80
#train4 : loanWords
to_check = [random.sample(range(nS), 50),random.sample(range(nS), 60),random.sample(range(nS), 24), 
            random.sample(range(nS), 8), random.sample(range(nS), 78),random.sample(range(nS), 20), 
            random.sample(range(nS), 40)]
"""           
to_check = [random.sample(range(20), 18),random.sample(range(20), 10),random.sample(range(20), 6), 
            random.sample(range(20), 2), random.sample(range(20), 18),random.sample(range(20), 4), 
            random.sample(range(20), 12),random.sample(range(20), 18), 
            random.sample(range(20), 18)]
"""
print(to_check)

X_dict = {}
kx = 6
kz = 9

for dS in ['.en','.ko']:
    for il in range(7):
        i = kz if il==0 or il>kx else il #i for call file, il for to_check
        fn = str(i)
        if i != kz:
            with open('en_es_data/train'+fn+dS,'r') as f:
                X = f.read().split('\n')[:-2][:1200000]
        else:
            if dS == '.en':
                X_dict[il] = get_loanwords(Xw)
                X = X_dict[il][0]
            else:
                X = X_dict[il][1]

        #X = [s for ik,s in enumerate(reversed(X)) if ik % 20 in to_check[il]]
        X = [s for ik,s in enumerate(X) if ik % nS in to_check[il]]
        print(dS, i, len(X), X[30][:50])
        save_op = 'w' if il == 0 else 'a'
        with open('en_es_data/train'+dS,save_op) as f:
            f.write('\n'.join(X))

to_check = [random.sample(range(80), 12)]
print(to_check)
for dS in ['.en','.ko']:
    with open('en_es_data/totrain'+dS, 'r') as f:  #train_mono_org'+dS,'r') as f:
        X = f.read().split('\n')[:-2]
    X = [s for ik,s in enumerate(X) if ik % 80 in to_check[0]]
    """
    if dS == '.ko':
        X = [p.sub(' ',q.sub('_','_ ' + s)) for ik,s in enumerate(reversed(X)) if ik % 80 in to_check[0]]
    else:
        X = [s for ik,s in enumerate(X) if ik % 80 in to_check[0]]
    """
    X = [s for s in X if len(s.split(' ')) > 2]
    print(dS, i, len(X), X[30][:50])
    with open('en_es_data/train_mono'+dS,'w') as f:
        f.write('\n'.join(X))