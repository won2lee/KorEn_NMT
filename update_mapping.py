import json
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from collections import Counter
from itertools import chain

#from nmt_model import NMT
from net_Module.net_model import Net

def net_run(embed, vocab, trained_mapping, nul_mapping, device):

    path = 'net_Module/'
    cutline = 0.20 # <= 0.17
    epsilon = 0.0001
    to_depreciate = 0.002 # <= 0.00005   

    #to run out of "run.py", load params first and call this function !!! 
    """
    params = torch.load('model.bin', map_location=lambda storage, loc: storage)
    embed = params['state_dict']['model_embeddings.vocabs.weight'].to("cuda:0")
    vocab = params['vocab']
    from update_mapping import net_run
    net_run(embed, vocab, "cuda:0")
    """

    #if dict_exists:
    with open(path+'dict_en.json', 'r') as f:
        bi_dict_en = json.load(f)
    with open(path+'dict_ko.json', 'r') as f:
        bi_dict_ko = json.load(f)
    #else:
    #bi_dict = {}

    #dictionary = torch.load(path + 'dictionary')

    net = Net(embed, vocab, device, 300)
    net = net.to(device)

    best = 0.1
    check = 0
    p = re.compile('[^a-zA-Z가-힣]')

    #if len(bi_dict_en) > 0:
    bi_dict = [bi_dict_en, bi_dict_ko]
    dictionary = []
    for bi_d in bi_dict:
        dictionary += list(chain(*[[(k,kv) for kv, vv in v.items() if vv > cutline] for k,v in bi_d.items()]))
    dictionary = list(set(dictionary))
    print(f"len of dictionary : {len(dictionary)}")

    n_iter = 2

    for i in range(n_iter):
        mappings = []
        distance = []
        added_w = []
        nul_mapping = nul_mapping if i==0 else True
        for il, slang in enumerate(['en','ko']):
            dictionary = dictionary if slang == 'en' else [(s[1],s[0]) for s in dictionary]
            W, m_cosine, dict_lang, scores = net(dictionary, trained_mapping[il], slang, nul_mapping=nul_mapping)
            to_plus = (0,54621) if slang == 'en' else (54621,0)
            dict_lang = dict_lang + torch.tensor(to_plus).to(device)

            bi_dict[il] = {k : {kv : vv - to_depreciate for kv,vv in v.items()} for k,v in bi_dict[il].items()} #to discount old_data
            new_dict = {net.vocs.id2word[w[0].item()]:(net.vocs.id2word[w[1].item()],scores[i].item()) 
                            for i,w in enumerate(dict_lang[:20000]) if scores[i].item() > 0.165 }
            bi_dict[il] = dict_update(new_dict, bi_dict[il])

            mappings.append(W)
            distance.append(m_cosine)
            added_w.append(sorted(new_dict.items(), key=lambda x:x[1][1])[:5])

        m_diff = torch.norm(torch.mm(mappings[0],mappings[1]) - torch.eye(300).to(device))
        if m_diff > 0.1:
            print("torch.mm(W[0],W[1] is not Identity Matrix. difference = {}".format(m_diff))
        #else:
            #print("torch.mm(W[0],W[1] is Identity Matrix. difference = {}".format(m_diff))

        dictionary = []
        for bi_d in bi_dict:
            dictionary += list(chain(*[[(k,kv) for kv, vv in v.items() if vv > cutline] for k,v in bi_d.items()]))
        dictionary = list(set(dictionary))
        
        """
        dict_en = {net.vocs[b[0]]:net.vocs[b[1]] for b in dictionary}
        dict_ko = {v:k for k,v in dict_en.items()}
        #dict_mapping = {i:i for i in range(len(net.vocs))}
        dict_mapping = {i:i if p.match(net.vocs.id2word[i]) is not None else 0 for i in range(len(net.vocs))}

        for dt in [dict_en,dict_ko]:
            dict_mapping.update(dt)  # dictionary 에 있는 단어는 번역 단어로 그렇지 않은 경우 그대로 ..
        print("dict mapping vocab의 갯수 : {}".format(len([k for k,v in dict_mapping.items() if v>0])))
        """
        if min(distance) > best + epsilon:
            best = min(distance)
            check = 0
        else:
            check += 1
            #print("check : {}".format(check))

        if check >1 or i == n_iter-1:

            for ik,fn in enumerate(['best_mapping_en.pth', 'best_mapping_ko.pth']):
                f_path = os.path.join(path, fn)
                #print('* Saving the mapping to %s ...' % f_path)
                torch.save(mappings[ik], f_path)

            torch.save(dictionary, path+'dictionary')

            for il, fn in enumerate(['dict_en.json', 'dict_ko.json']):
                with open(path+fn, 'w') as f:
                    json.dump(bi_dict[il],f)

            print(len(dictionary), added_w)  #dictionary[-5:])
            #print("n_iter : {},  check : {},  STOP Iteration".format(i,check))            
            break

    dictionary2 = []
    for bi_d in bi_dict:
        dictionary2 += list(chain(*[[(k,kv) for kv, vv in v.items() if vv > -0.2] for k,v in bi_d.items()]))
    dictionary2 = list(set(dictionary2))

    #dict_en = make_rand_dict([(net.vocs[b[0]],net.vocs[b[1]]) for b in dictionary2])
    dict_en = make_uniq_dict([(net.vocs[b[0]],net.vocs[b[1]]) for b in dictionary2])
    print("len(dictionary2) : {},len(dict_en) : {}".format(len(dictionary2),len(dict_en)))
    #dict_ko = make_rand_dict([(net.vocs[b[1]],net.vocs[b[0]]) for b in dictionary2])
    dict_ko = make_uniq_dict([(net.vocs[b[1]],net.vocs[b[0]]) for b in dictionary2])
    #dict_mapping = {i:i for i in range(len(net.vocs))}
    # if map learning -i not map learning 0
    dict_mapping = {i:i if p.match(net.vocs.id2word[i]) is not None else -i for i in range(len(net.vocs))}
    #dict_mapping = {i:i if p.match(net.vocs.id2word[i]) is not None else 0 for i in range(len(net.vocs))}

    for dt in [dict_ko,dict_en]:
        dict_mapping.update(dt)  # dictionary 에 있는 단어는 번역 단어로 그렇지 않은 경우 그대로 ..
    print("dict mapping vocab의 갯수 : {}".format(len([k for k,v in dict_mapping.items() if v>0])))
    

    return mappings[0], mappings[1], dict_mapping

def make_uniq_dict(tuple_L):
    rand_dict = {}
    for (x,y) in tuple_L:
        if x in rand_dict.keys():
            if rand_dict[x] != y and rand_dict[x] != "not_unique":
                rand_dict[x] = "not_unique"
        else:
            rand_dict[x] = y
    
    rand_dict = {k:v for k,v in rand_dict.items() if v!="not_unique"}

    return rand_dict  

def make_rand_dict(tuple_L):
    rand_dict = {}
    for (x,y) in tuple_L:
        if x in rand_dict.keys():
            if np.random.randint(3) >0:
                continue
            else:
                rand_dict[x] = y
        else:
            rand_dict[x] = y
    return rand_dict

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