

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


#train4 : loanWords
to_check = [random.sample(range(20), 18),random.sample(range(20), 12),random.sample(range(20), 5), 
            random.sample(range(20), 3), random.sample(range(20), 18),random.sample(range(20), 18), 
            random.sample(range(20), 18)]
print(to_check)

X_dict = {}
for dS in ['.en','.ko']:
    for il in range(7):
        i = 4 if il==0 or il>4 else il #i for call file, il for to_check
        fn = str(i)
        if i != 4:
            with open('en_es_data/train'+fn+dS,'r') as f:
                X = f.read().split('\n')[:1200000]
        else:
            if dS == '.en':
                X_dict[il] = get_loanwords(Xw)
                X = X_dict[il][0]
            else:
                X = X_dict[il][1]

        #X = [s for ik,s in enumerate(reversed(X)) if ik % 20 in to_check[il]]
        X = [s for ik,s in enumerate(X) if ik % 20 in to_check[il]]
        print(dS, i, len(X), X[30][:50])
        save_op = 'w' if il == 0 else 'a'
        with open('en_es_data/train'+dS,save_op) as f:
            f.write('\n'.join(X))

to_check = [random.sample(range(80), 15)]
print(to_check)
for dS in ['.en','.ko']:
    with open('en_es_data/train_mono_org'+dS,'r') as f:
        X = f.read().split('\n')[:-2]
    if dS == '.ko':
        X = [p.sub(' ',q.sub('_','_ ' + s)) for ik,s in enumerate(reversed(X)) if ik % 40 in to_check[0]]
    else:
        X = [s for ik,s in enumerate(X) if ik % 40 in to_check[0]]
    print(dS, i, len(X), X[30][:50])
    with open('en_es_data/train_mono'+dS,'w') as f:
        f.write('\n'.join(X))


import numpy as np

import random

#train4 : loanWords
to_check = [random.sample(range(20), 15),random.sample(range(20), 6), random.sample(range(20), 4), random.sample(range(20), 6)]
print(to_check)
for dS in ['.en','.ko']:
    for i in range(1,5):
        fn = str(i)
        with open('en_es_data/train'+fn+dS,'r') as f:
            X = f.read().split('\n')[:1200000]

        X = [s for ik,s in enumerate(X) if ik % 20 in to_check[i-1]]
        print(dS, i, len(X))
        save_op = 'w' if i == 1 else 'a'
        with open('en_es_data/train'+dS,save_op) as f:
            f.write('\n'.join(X))

to_check = [np.random.randint(0,20,7), np.random.randint(0,20,4)]
for dS in ['.en','.ko']:
    for i in range(1,4):
        fn = str(i)
        with open('en_es_data/train'+fn+dS,'r') as f:
            X = f.read().split('\n')[:1200000]
        if i>1:
            X = [s for ik,s in enumerate(X) if ik % 20 in to_check[i-2]]
        print(dS, i, len(X))
        save_op = 'w' if i == 1 else 'a'
        with open('en_es_data/train'+dS,save_op) as f:
            f.write('\n'.join(X))


for dS in ['.en','.ko']:
    for i in range(1,5):
        fn = str(i) if i !=4 else '_vocab'
        with open('en_es_data/train'+fn+dS,'r') as f:
            X = f.read().split('\n')[:1200000]
        print(dS, i, len(X))
        save_op = 'w' if i == 1 else 'a'
        with open('en_es_data/train'+dS,save_op) as f:
            f.write('\n'.join(X))



import random
import numpy as np
import re
from tqdm.notebook import tqdm
import json

with open('en_es_data/loanW_modi.json','r') as f:
    loanW = json.load(f)
ePN = loanW['ePN']
eRN = loanW['eRN']
xPN = loanW['xPN']
xRN = loanW['xRN']
print([len(loanW[d]) for d in "ePN eRN xPN xRN".split(' ')])

p= re.compile('[^a-z\^\_\s\`\-\&\:\.\“\”]')

nS = [5000, 5000, 500, 500]
enN = []
koN = []
for i,ds in enumerate([ePN,eRN,xPN, xRN]):
    X = [w for w in ds if len(p.findall(w[0])) <3]
    print(len(X))
    nT = list(range(len(X)))
    for ik in tqdm(range(nS[i])):
        nr = np.random.randint(6,16)
        to_get = random.sample(nT, nr)
        enN.append(','.join([X[k][0] for k in to_get]))
        koN.append(','.join([X[k][1] for k in to_get])) 

with open('en_es_data/loanW.en','w') as f:
    f.write('\n'.join(enN))
with open('en_es_data/loanW.ko','w') as f:
    f.write('\n'.join(koN))

print(len(enN),enN[10],'\n',koN[10])


import re 
p = re.compile('\_\s*원문\s*기사\s*보기')
with open('en_es_data/temp_train1/train1.ko','r') as f:
    X = f.read().split('\n')
with open('en_es_data/temp_train1/train1.en','r') as f:
    en = f.read().split('\n')
X = [p.sub('',s).strip() for s in X]
#중복기사 지우기
new_ko = [X[0]]
new_en = [en[0]]
for i,s in enumerate(X):
    if s == new_ko[-1] and en[i] == new_en[-1]:
        continue
    new_ko.append(s)
    new_en.append(en[i])
fs = [new_en,new_ko]
for il,lg in enumerate(['en','ko']):
    with open('en_es_data/train1.'+lg,'w') as f:
        f.write('\n'.join(fs[il]))

import re 
p = re.compile('\_\s*원문\s*기사\s*보기')
with open('en_es_data/temp_train1/dev.ko','r') as f:
    X = f.read().split('\n')
X = [p.sub('',s).strip() for s in X]
with open('en_es_data/dev.ko','w') as f:
    f.write('\n'.join(X))
