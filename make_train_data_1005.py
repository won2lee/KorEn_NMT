# 1 : 일상적인 데이터 생성
# 2 : copy backtranslated sentences for self tanslation learning
# 3 : 카피된 데이터 확인
# 4 : mono corpus data 추가 처리 ( _ _ 이중 어절 나누기 등 삭제) 

################################################################################
# 1 : 일상적인 데이터 생성

import random
import re
import numpy as np

def get_loanwords(Xw):
    Xnew =[[],[]]
    for i,s in enumerate(Xw[0]):
        index_array = list(range(len(s)))
        np.random.shuffle(index_array)
        Xnew[0].append(p.sub(' ',' _ . '.join([s[idx] for idx in index_array]).strip()))
        s2 = Xw[1][i]
        Xnew[1].append(p.sub(' ',' _ . '.join([s2[idx] for idx in index_array]).strip()))
    return Xnew

p = re.compile('\s+')
q = re.compile('\_\s*\_')
z = re.compile('\s*\_\s*$')
z1 = re.compile('\s*\_\s*\.\s*$')

path = "../drive/My Drive/en_es_data/"

#sbol = ['_','^','`']

Xw = [] #loan_words
for i,dS in enumerate(['.en','.ko']):
    with open(path + 'train4_before'+dS,'r') as f:
        Xw.append(f.read().split('\n')[:1200000])

Xw = [[s.split(',') for s in Xw[i]] for i in range(2)]

nS = 800
xch = 1 # if normal 1 elif ft =1
#train4 : loanWords

to_check1 = [random.sample(range(nS), 700),random.sample(range(nS), 700),random.sample(range(nS), 200), 
            random.sample(range(nS), 80), random.sample(range(nS), 700),random.sample(range(nS), 150), 
            random.sample(range(nS), 200),random.sample(range(nS), 700),random.sample(range(nS), 500),
            random.sample(range(nS), 300)]  ## 2,3,5,6,7 +=> 2배 

to_check2 = [random.sample(range(nS), 7),random.sample(range(nS), 700),random.sample(range(nS), 75), 
            random.sample(range(nS), 20), random.sample(range(nS), 700),random.sample(range(nS), 50), 
            random.sample(range(nS), 100),random.sample(range(nS), 700),random.sample(range(nS), 7),
            random.sample(range(nS), 7)]  ## 2,3,5,6,7 +=> 2배 


def to_make_train(dstn, to_check,z,z1):
    X_dict = {}
    kx = 7
    kz = 10
    lenD= []
    print(to_check)
    for dS in ['.en','.ko']:
        lenD.append(0)
        for il in range(10):
            i = kz if il==0 or il>kx else il #i for call file, il for to_check
            fn = str(i)
            if i != kz:
                with open(path + 'train'+fn+dS,'r') as f:
                    X = f.read().split('\n')[:-2]
                    X = X[23950:27202]*2 if i == 7 else X[:1200000]
            else:
                if dS == '.en':
                    X_dict[il] = get_loanwords(Xw)
                    X = X_dict[il][0]
                else:
                    X = X_dict[il][1]

            #X = [s for ik,s in enumerate(reversed(X)) if ik % 20 in to_check[il]]
            X = [z1.sub('',z.sub('',s)) if ik % 11 ==0 else z.sub('',s) for ik,s in enumerate(X) if ik % nS in to_check[il]]
            print(dS, i, len(X), X[30][:50])
            save_op = 'w' if il == 0 else 'a'
            with open(dstn+dS,save_op) as f:
                f.write('\n'.join(X))
            lenD[-1]+= len(X)
            
    return lenD

lenD = to_make_train('en_es_data/train', to_check1, z, z1)
print(lenD)
lenD = to_make_train('en_es_data/train_st', to_check2, z, z1)
print(lenD)


Xbi={}
lenT = 50000
for dS in ['.en','.ko']:
    with open('en_es_data/train_st'+dS, 'r') as f:
         X = f.read().split('\n')
    index_array = list(range(len(X)))
    np.random.shuffle(index_array)
    Xbi[dS] = [X[i] for i in index_array[:lenT]]

to_check = [random.sample(range(80), 5),random.sample(range(80), 10)] #45)]

print(to_check)
for dS in ['.en','.ko']:
    with open(path +'totrain1'+dS, 'r') as f:  #train_mono_org'+dS,'r') as f:
        X = f.read().split('\n')[:-2]   
    X = [s for ik,s in enumerate(X) if ik % 80 in to_check[0]]
    print(len(X))
    with open(path +'totrain2'+dS, 'r',encoding="utf8", errors='ignore') as f:  #train_mono_org'+dS,'r') as f:
        X2= f.read().split('\n')[:-2]   
    X += [s for ik,s in enumerate(X2) if ik % 80 in to_check[1]]
    X += Xbi[dS]
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
        
        
###################################################
# 2 : copy backtranslated sentences 


# copy backtranslated sentences 
#     for self tanslation learning
#     (select 최신 nK*valid_niter*train_batch_size  )


nK = 20
valid_niter = 1000
train_batch_size = 20
for ik,sl in enumerate(['ko2en', 'en2ko']):
    with open('en_es_data/pseudo_bi_src/en_src' if ik==0 else 'en_es_data/pseudo_bi_src/ko_src','r') as f:
        biY = f.read().split('\n')[-nK*valid_niter*train_batch_size:]
    with open('en_es_data/pseudo_bi_src/ko_made' if ik==0 else 'en_es_data/pseudo_bi_src/en_made','r') as f:
        biX = f.read().split('\n')[-nK*valid_niter*train_batch_size:]

    with open('en_es_data/pseudo_bi/en_src' if ik==0 else 'en_es_data/pseudo_bi/ko_src','w') as f:
        f.write('\n'.join(biY))
    with open('en_es_data/pseudo_bi/ko_made' if ik==0 else 'en_es_data/pseudo_bi/en_made','w') as f:
        f.write('\n'.join(biX))

# copy backtranslated sentences 
# for back tanslation learning  
# (random selection)
        
bs = 80
to_check = random.sample(range(bs), 40)
print(to_check)
kk = 0 #-200000
with open('en_es_data/pseudo_bi_src/en_made','r') as f:
    en_t = f.read().split('\n')[kk:]
en_t = [s for i,s in enumerate(en_t[kk:]) if i%bs in to_check]
with open('en_es_data/pseudo_bi/en_made','w') as f:
    f.write('\n'.join(en_t))
with open('en_es_data/pseudo_bi_src/ko_src','r') as f:
    ko_s = f.read().split('\n')[kk:]
ko_s = [s for i,s in enumerate(ko_s[kk:]) if i%bs in to_check]
with open('en_es_data/pseudo_bi/ko_src','w') as f:
    f.write('\n'.join(ko_s))

with open('en_es_data/pseudo_bi_src/ko_made','r') as f:
    ko_t = f.read().split('\n')[kk:]
ko_t = [s for i,s in enumerate(ko_t[kk:]) if i%bs in to_check]
with open('en_es_data/pseudo_bi/ko_made','w') as f:
    f.write('\n'.join(ko_t))
with open('en_es_data/pseudo_bi_src/en_src','r') as f:
    en_s = f.read().split('\n')[kk:]
en_s = [s for i,s in enumerate(en_s[kk:]) if i%bs in to_check]
with open('en_es_data/pseudo_bi/en_src','w') as f:
    f.write('\n'.join(en_s))
    

#################################################################
# 3: 카피된 데이터 확인

with open('en_es_data/pseudo_bi/en_made','r') as f:
    en_t = f.read().split('\n')

with open('en_es_data/pseudo_bi/ko_src','r') as f:
    ko_s = f.read().split('\n')


with open('en_es_data/pseudo_bi/ko_made','r') as f:
    ko_t = f.read().split('\n')

with open('en_es_data/pseudo_bi/en_src','r') as f:
    en_s = f.read().split('\n')

len(en_t), len(ko_s), len(ko_t), len(en_s)


##################################################################
# 4 : mono corpus data 추가 처리 ( _ _ 이중 어절 나누기 등 삭제) 

import re
path2 = '../drive/My Drive/en_es_data/'
path = 'en_es_data/'
p = re.compile('[a-z가-힣0-9\,]')
p1 = re.compile('\_\s+(?P<t1>[\_\^\`])')
p2 = re.compile('\s*\_\s*$')
sbol = ['_','^']
XX = [[],[]]
cline = [0.7,0.6]
for ik,dS in enumerate(['.en','.ko']):
    with open(path2 +'totrain2'+dS, 'r',encoding="utf8", errors='ignore') as f:  #train_mono_org'+dS,'r') as f:
        X = f.read().split('\n')[:-2]
    #X = [s for ik,s in enumerate(X) if ik % 80 in to_check[0]]
    XX[ik] = []
    for s in X:
        s = p1.sub('\g<t1>',p2.sub('',s))
        ss = [w for w in s.split(' ') if w not in sbol]
        wR = len([w for w in ss if p.search(w) is not None]) / len(ss)
        if wR>cline[ik] and len(s.split(' '))<150:
            XX[ik].append(s) 

        #XX.append((len([w for w in ss if p.search(w) is not None]), len(ss)))
    #print(sum([1 if z[0]/z[1]>0.8 else 0 for z in XX]), len(XX))
    print(len(XX[ik]), len(X))

    with open('en_es_data/totrain2'+dS, 'w') as f:  #train_mono_org'+dS,'r') as f:
        f.write('\n'.join(XX[ik]))

    #xx = len([w len([w for w in s.split(' ') if w not in sbol]) for s in X]
