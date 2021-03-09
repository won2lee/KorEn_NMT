import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from collections import Counter
from itertools import chain


path = 'net_Module/'
device = "cuda:0"
dict_exists = True
cutline = 0.17
epsilon = 0.001
"""
with open(path+'dict_enT.json','r') as f:
    dictE = json.load(f)
with open(path+'dict_koT.json','r') as f:
    dictK = json.load(f)
"""    
from nmt_model import NMT
from net_Module.net_model import Net

params = torch.load('model.bin', map_location=lambda storage, loc: storage)
#args = params['args']
#model = NMT(vocab=params['vocab'], **args)
#model.load_state_dict(params['state_dict'])

#torch.save(params['state_dict'], 'model_bi_0714')

#slang = 'en'

#print("slang is {}".format(slang))

#dictEK = [dictE, dictK]

path = 'net_Module/'
#fpath = path+'dict_en.json' if slang == 'en' else path+'dict_ko.json'
if dict_exists:
    with open(path+'dict_en.json', 'r') as f:
        bi_dict_en = json.load(f)
    with open(path+'dict_ko.json', 'r') as f:
        bi_dict_ko = json.load(f)
else:
    bi_dict = {}
    #fpath = path+'dictionary_en' if slang == 'en' else path+'dictionary_ko'
    dictionary = torch.load(path + 'dictionary')
"""
k = 2 if slang =='en' else 1
bi_words = [(w,sorted(v, key=lambda x:x[1][0], reverse = True)[0][0]) for w,v in dictEK[k%2].items()]
bi_words += [(sorted(v, key=lambda x:x[1][0], reverse = True)[0][0],w) for w,v in dictEK[(k+1)%2].items()]
bi_words = list(set(bi_words))
dictionary = bi_words
"""
net = Net(params,300)
net = net.to(device)

best = 0.4
patience = 0
to_depriciate = 0.002
#dict_count = {}

def dict_merge(d1,d2):
    for k,v in d1.items():
        if k in d2.keys():
            d2[k] += v
        else:
            d2[k] = v

def dict_update(d1,d2):  # d1's value : tuple, d2's value : dict
    for k,v in d1.items():
        if k not in d2.keys():
            d2[k] = {v[0]:v[1]}
        elif v[0] not in d2[k].keys():
            d2[k][v[0]] = v[1]
        else:
            d2[k][v[0]] = max(v[1],d2[k][v[0]])
    return d2

#dictionary = torch.tensor([(net.vocs[b[0]], net.vocs[b[1]]) for b in bi_words[:5000]])
#torch.save(dictionary, path+'dictionary_en' if slang =='en' else path+'dictionary_ko' )
if len(bi_dict_en) > 0:
    dictionary = list(chain(*[[(k,kv) for kv, vv in v.items() if vv > cutline] for k,v in bi_dict_en.items()]))
    dictionary += list(chain(*[[(kv,k) for kv, vv in v.items() if vv > cutline] for k,v in bi_dict_ko.items()]))
    dictionary = list(set(dictionary))
    #dictionary = torch.tensor(list(set(dictionary))).to(device)
#count = Counter([(w[0],w[1]) for w in dictionary.numpy()])
#dict_merge(count,dict_count)
n_iter = 2
bi_dict = [bi_dict_en, bi_dict_ko]
for i in range(n_iter):
    mapping = []
    distance = []
    for il, slang in enumerate(['en','ko']):
        dictionary = dictionary if slang == 'en' else [(s[1],s[0]) for s in dictionary]#dictionary[:,[1,0]]
        W, m_cosine, dict_lang, scores = net(dictionary, slang)
        to_plus = (0,54621) if slang == 'en' else (54621,0)
        dict_lang = dict_lang + torch.tensor(to_plus).to(device)

        bi_dict[il] = {k : {kv : vv - to_depriciate for kv,vv in v.items()} for k,v in bi_dict[il].items()} #to discount old_data
        new_dict = {net.vocs.id2word[w[0].item()]:(net.vocs.id2word[w[1].item()],scores[i].item()) 
                        for i,w in enumerate(dict_lang[:10000]) if scores[i].item() > 0.15 }
        bi_dict[il] = dict_update(new_dict, bi_dict[il])

        mapping.append(W)
        distance.append(m_cosine)
    
    m_diff = np.linalg.norm((torch.mm(mapping[0],mapping[1]) - torch.eye(300).to(device)).cpu())
    if m_diff > 1.0:
        print("torch.mm(W[0],W[1] is not Identity Matrix. difference = {}".format(m_diff))
    else:
        print("torch.mm(W[0],W[1] is Identity Matrix. difference = {}".format(m_diff))

    dictionary = list(chain(*[[(k,kv) for kv, vv in v.items() if vv > cutline] for k,v in bi_dict_en.items()]))
    dictionary += list(chain(*[[(kv,k) for kv, vv in v.items() if vv > cutline] for k,v in bi_dict_ko.items()]))
    dictionary = list(set(dictionary))
    #dictionary = torch.tensor(list(set(dictionary))).to(device)
    print(len(dictionary), dictionary[-5:])

    if min(distance) > best + epsilon:
        best = min(distance)
        patience = 0

        torch.save(dictionary, path+'dictionary')

        for il, fn in enumerate(['dict_en.json', 'dict_ko.json']):
            with open(path+fn, 'w') as f:
                json.dump(bi_dict[il],f)
    else:
        patience += 1
        print("patience : {}".format(patience))

    if patience >1 or i == n_iter-1:
        print("n_iter : {},  patience : {},  STOP Iteration".format(i,patience))            
        break


"""
    dictionary = list(chain(*[[(k,kv) for kv, vv in v.items() if vv > cutline ] for k,v in bi_dict.items()]))
    print(len(dictionary), dictionary[-5:])

    torch.save(dictionary, path+'dictionary_en' if slang =='en' else path+'dictionary_ko')

    with open(path+'dict_en.json' if slang =='en' else path+'dict_ko.json', 'w') as f:
        json.dump(bi_dict,f)
    
    if patience >10 or i == n_iter-1:
            print("n_iter : {},  patience : {},  STOP Iteration".format(i,patience))            
            break
    

    #pre_dic = dictionary    
    W, m_cosine, dictionary, scores = net(dictionary, slang)

    if slang == 'en':
        dictionary = dictionary + torch.tensor((0,54621)).to("cuda:0")
    else:
        dictionary = dictionary + torch.tensor((54621,0)).to("cuda:0")
    print("iteration {}, m_cosine = {}".format(i,m_cosine))
            
    #dict_count = Counter(chain(*[[(w[0],w[1]) for w in dct.numpy()] for dct in [pre_dic, dictionary[:3000]]]))
    #count = Counter([(w[0],w[1]) for w in dictionary[:5000].numpy()])
    #dict_merge(count,dict_count)
    #dictionary = torch.tensor([w[0] for w in sorted(dict_count.items(), key=lambda x:x[1], reverse=True) if w[1]>min(i/2,2)])
    #dict_all = dictionary
    #dictionary = dictionary[:5000]
    
    if dict_exists:
        bi_dict = {k : {kv : vv - to_depriciate for kv,vv in v.items()} for k,v in bi_dict.items()} #to discount old_data
    
    new_dict = {net.vocs.id2word[w[0].item()]:(net.vocs.id2word[w[1].item()],scores[i].item()) 
                    for i,w in enumerate(dictionary[:10000]) if scores[i].item() > 0.15 }
    bi_dict = dict_update(new_dict, bi_dict)
    #print([(k,bi_dict[k]) for k in list(bi_dict.keys())[:10]])

    if m_cosine > best:
        best = m_cosine
        patience = 0
    else:
        patience += 1
        print("patience : {}".format(patience))

    dictionary = list(chain(*[[(k,kv) for kv, vv in v.items() if vv > cutline ] for k,v in bi_dict.items()]))
    print(len(dictionary), dictionary[-5:])

    torch.save(dictionary, path+'dictionary_en' if slang =='en' else path+'dictionary_ko')

    with open(path+'dic_en.json' if slang =='en' else path+'dict_ko.json', 'w') as f:
        json.dump(bi_dict,f)
    
    if patience >10 or i == n_iter-1:
            print("n_iter : {},  patience : {},  STOP Iteration".format(i,patience))            
            break

"""
