#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skeleton of this code is based on 
Stanford CS224N 2018-19: Homework 4 (code lines : )
however about 2/3 of the code is modified and added (code lines : )

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --mono-en=<file> --mono-ko=<file> --vocab=<file> --map-en=<file> --map-ko=<file> --slang=<str> --tlang=<str> --mapping=<int>  --map_learning=<int> --self_trans=<int> --back_trans=<int> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE --slang=<str> --tlang=<str> 
    run.py backtr [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE --slang=<str> --tlang=<str> --mapping=<int> --batch-size=<int>

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --mono_en=<file>                        mono english file
    --mono_ko=<file>                        mono korean file
    --vocab=<file>                          vocab file
    --map-en=<file>                         mapping en file
    --map-ko=<file>                         mapping ko file
    --slang=<str>                           source language 
    --tlang=<str>                           target language 
    --batch_ratio=<int>                     batch_ratio [default: 1]
    --loss_ratio=<int>                      loss_ratio [default: 1]
    --self_learning=<int>                   self_learning [default: 0]
    --mapping=<int>                         mapping [default: 0]
    --map_learning=<int>                    map_learning [default: 0]
    --self_trans=<int>                      self_trans [default: 0] 
    --back_trans=<int>                      back_trans [default: 0] 
    --batch_plus=<int>                      batch_plus [default: 0]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]  
    --embed-size=<int>                      embedding size [default: 128]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 3.0]
    --log-every=<int>                       log every [default: 100]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 10]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.0010]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 1000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 100]
"""
import math
import sys
import pickle
import time
import shutil


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Khypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter, sents_ordering
from vocab import Vocab, VocabEntry, generate_wid2cid, get_wid2cid
from update_mapping import net_run

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, slang, tlang, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size, slang, tlang):
            loss = -model(src_sents, tgt_sents, slang=slang, tlang=tlang).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references, hypotheses): #: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    ##########################
    train_data_src = read_corpus(args['--train-tgt'], source='src')
    train_data_tgt = read_corpus(args['--train-src'], source='tgt')
    train_data_reverse = list(zip(train_data_src, train_data_tgt)) 
    """
    train_data_src = read_corpus(args['--mono-en'], source='src')
    train_data_tgt = read_corpus(args['--mono-en'], source='tgt')
    train_data_eng = list(zip(train_data_src, train_data_tgt)) 

    train_data_src = read_corpus(args['--mono-ko'], source='src')
    train_data_tgt = read_corpus(args['--mono-ko'], source='tgt')
    train_data_kor = list(zip(train_data_src, train_data_tgt)) 
    """
    #########################   

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])
    #wid2cid, char_size = generate_wid2cid()
    #print("char_size : {}".format(char_size))
    wid2cid = get_wid2cid()
    char_size = 85
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                char_size = char_size,
                vocab=vocab,
                wid2cid = wid2cid)
    
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    #vocab_mask = torch.ones(len(vocab.tgt))
    #vocab_mask[vocab.tgt['<pad>']] = 0
    vocab_mask = torch.ones(len(vocab.vocs))
    vocab_mask[vocab.vocs['<pad>']] = 0
    
    #device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = num_trial = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    
    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
    hist_valid_scores = [-14.175984, -8.718, -6.898558, -6.249343] # after 42000 iteration ,-6.264680]   
    """
    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)
    params2 = torch.load(model_save_path + '.optim')
    l_rate = 0.0001
    params2['param_groups'][0]['lr'] = l_rate
    optimizer.load_state_dict(params2)
    hist_valid_scores = [-3.515570, -3.509147, -3.508350]
    """

    slang=args['--slang']
    tlang=args['--tlang']

    reverse_batch = batch_iter(train_data_reverse, train_batch_size, tlang, slang, shuffle=True) 
    #forward_batch = batch_iter(train_data, train_batch_size, slang, tlang, shuffle=True)

    b_ratio = int(args['--batch_ratio'])
    l_ratio = int(args['--loss_ratio'])
    self_learning_X = int(args['--self_learning'])
    mapping_X = int(args['--mapping'])
    map_learning = int(args['--map_learning'])
    back_trans_X = int(args['--back_trans'])
    batch_plus = int(args['--batch_plus'])
    patience_ceiling = int(args['--patience'])
    self_trans_X = int(args['--self_trans'])

    pseudo_load = 1
    st_batch = 0
    bt_batch = 0

    mapping = 0
    cut_line = -4.60
    c_plus = 0.00025
    nK = 10 #12

    n_save = 0   # colab 의 갑작스런 다운에 대비한 파일 저장을 위해 ...

    """
    #bi_ratio self_ratio map_ratio
    if bitext_learning:
        enko_batch = batch_iter(train_data, train_batch_size//bi_ratio, slang, tlang, shuffle=True)
        koen_batch = batch_iter(train_data_reverse, train_batch_size//bi_ratio, tlang, slang, shuffle=True) 
        #eng_batch = batch_iter(train_data_eng, train_batch_size//b_ratio, slang, slang, shuffle=True) 
        #kor_batch = batch_iter(train_data_kor, train_batch_size//b_ratio, tlang, tlang, shuffle=True)      
    """
    if self_learning_X==1 or mapping_X==1 or ((back_trans_X==1 or self_trans_X ==1) and pseudo_load ==0):
        train_data_src = read_corpus(args['--mono-en'], source='src')
        train_data_tgt = read_corpus(args['--mono-en'], source='tgt')
        train_data_eng = list(zip(train_data_src, train_data_tgt)) 

        train_data_src = read_corpus(args['--mono-ko'], source='src')
        train_data_tgt = read_corpus(args['--mono-ko'], source='tgt')
        train_data_kor = list(zip(train_data_src, train_data_tgt)) 

    if self_learning_X==1:
        eng_batch = batch_iter(train_data_eng, train_batch_size//b_ratio, slang, slang, shuffle=True) 
        kor_batch = batch_iter(train_data_kor, train_batch_size//b_ratio, tlang, tlang, shuffle=True)
   
    if mapping_X==1:
        #map_en = torch.load(args['--map-en']).to(device)
        #map_ko = torch.load(args['--map-ko']).to(device)
        eng_mapping_batch = batch_iter(train_data_eng, train_batch_size//b_ratio, slang, slang, shuffle=True, mapping=True) 
        kor_mapping_batch = batch_iter(train_data_kor, train_batch_size//b_ratio, tlang, tlang, shuffle=True, mapping=True)

    if (back_trans_X==1 or self_trans_X ==1) and pseudo_load ==0:
        en_back_batch = batch_iter(train_data_eng, train_batch_size//b_ratio + batch_plus, slang, tlang, shuffle=True) 
        ko_back_batch = batch_iter(train_data_kor, train_batch_size//b_ratio + batch_plus, tlang, slang, shuffle=True)


    if self_trans_X ==1 and pseudo_load ==1:
        s_trns = {}
        k_div = 1
        for i,dx in enumerate([['en','ko'],['ko','en']]):
            p_data_src = read_corpus('en_es_data/pseudo_bi/'+dx[0]+'_src', source='src')
            p_data_tgt = read_corpus('en_es_data/pseudo_bi/'+dx[1]+'_made', source='tgt')
            train_data_p = list(zip(p_data_src, p_data_tgt)) 
            s_trns[dx[0]+'2'+dx[1]] = batch_iter(train_data_p, train_batch_size, dx[0], dx[1], shuffle=True, self_trns =1) 


    if back_trans_X ==1 and pseudo_load ==1:
        b_trns = {}
        k_div = 5
        bt_batch = nK*k_div
        for i,dx in enumerate([['en','ko'],['ko','en']]):
            p_data_src = read_corpus('en_es_data/pseudo_bi/'+dx[0]+'_made', source='src')
            p_data_tgt = read_corpus('en_es_data/pseudo_bi/'+dx[1]+'_src', source='tgt')
            train_data_p = list(zip(p_data_src, p_data_tgt)) 
            b_trns[dx[0]+'2'+dx[1]] = batch_iter(train_data_p, train_batch_size//k_div, dx[0], dx[1], shuffle=True) 

    langs = [slang, tlang]

    #################  

    print("slang : {}, tlang : {}".format(slang, tlang))

    n_cycle = 6*nK
    valid_metric = 0.


    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, train_batch_size, slang, tlang, shuffle=True):
        #for i_of_iter in range(len(train_data)//train_batch_size):
            train_iter += 1

            #if train_iter > 20000:
            #    print('reached max iteration !!', file=sys.stderr)
            #    exit(0)
            """
            if train_iter != 1 and train_iter % (n_cycle * valid_niter) == 1 and valid_metric < max(hist_valid_scores):
                
                params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(device)
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

            
            if  train_iter != 1 and train_iter % valid_niter == 1 and patience > 10:
                l_rate *= 1.
                patience = 0
                print(f"learning rate is lowered to {l_rate}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = l_rate
            """
            

            if train_iter % valid_niter ==1:

                n_X = ((train_iter-1) % (n_cycle * valid_niter))//valid_niter

                if self_learning_X==1: 
                    self_learning=0 if train_iter % 1200 > 400 else 1
                else:
                    self_learning =0

                if mapping_X==1:
                    #mapping = 1
                    mapping=1 if n_X in [0,1,2,3,4,5,6,7,8,9] and valid_metric>-3.485 else 0
                else:
                    mapping =0

                if self_trans_X==1: 
                    #self_trans=0 if (train_iter-1) % (4 * valid_niter) > nK * valid_niter-1 else 1
                    #self_trans=1 if (n_X in list(range(nK)) and valid_metric>max(hist_valid_scores)-0.04) else 0    #  > nK * valid_niter-1 else 1
                    self_trans=1 if n_X in list(range(nK)) else 0    #  > nK * valid_niter-1 else 1
                    #,7,8,9,10,11,12,13,14,15,16,17,18,19
                else:
                    self_trans = 0
                """
                if self_trans_X==1:
                    if back_trans ==0:
                        self_trans=0 if (train_iter-1) % (2 * valid_niter) > 1*valid_niter-1 else 1
                    else:
                        self_trans=1
                else:
                    self_trans = 0
                """
                if back_trans_X ==1:
                    back_trans = 1 if n_X in list(range(5*nK)) and mapping==0 and self_trans==0 else 0 #and bt_batch>0 else 0
                    
                    #back_trans = 1 if n_X in list(range(nK,6*nK)) and mapping==0 and self_trans==0 else 0 #and bt_batch>0 else 0
                    #back_trans = 1
                    """
                    if self_trans_X ==1:                        
                        back_trans=0 if ((train_iter-1) % (4 * valid_niter) > 2*nK*valid_niter-1 or self_trans == 1) else 1
                    else:
                        back_trans =1
                    """
                else:
                    back_trans = 0

            #if train_iter > valid_niter and valid_metric < cut_line:
            #    self_trans = 0 # back_trans=0

            """
            if train_iter % 100 == 1:
                t_f = False if self_trans == 1 else True
                model.ko_decoder.requires_grad = t_f
                model.en_decoder.requires_grad = t_f
                model.en_att_projection.weight.requires_grad = t_f
                model.ko_att_projection.weight.requires_grad = t_f       
                model.en_combined_output_projection.weight.requires_grad = t_f
                model.ko_combined_output_projection.weight.requires_grad = t_f
                model.target_vocab_projection.weight.requires_grad = t_f
            """
            
            #if (back_trans==1 or self_trans==1) and train_iter == 1: #% valid_niter == 1: 
            if (back_trans==1 or self_trans==1) and pseudo_load==0 and train_iter % valid_niter == 1: 
                if (back_trans ==1 and bt_batch < 1) or (self_trans ==1 and st_batch ==0): 
                    b_trns = {}
                    s_trns = {}
                    b_gen = [en_back_batch, ko_back_batch]
                    cutlines = []
                    sbol = ['_','^','`']
                    sl2 = ['en2ko','ko2en']
                    k_div = 1 #### if back_trans==1 and self_trans==1 else 2 #1
                    k_plus = 1 if back_trans_X == 1 else 4

                    print("start back-translation ................")
                    for inK in range(nK):
                        print(f"back_trans step {inK} started")
                        for ik,sl in enumerate(['ko2en', 'en2ko']):
                            lenR = 0.55 if ik==0 else 0.70
                            back_src_sents = []
                            for i in range((valid_niter//k_div+valid_niter//(k_div*k_plus))*1):
                                back_src_sents += next(b_gen[ik])[0]
                            #back_src_sents = [s for s in back_src_sents if len(s) > 1]
                            nmted_sents, sent_scores = greedy_search(model, back_src_sents, batch_size=train_batch_size * 40,
                                        max_decoding_time_step= 100, slang = langs[ik%2], tlang = langs[(ik+1)%2], mapping = 0)
                            #print( langs[ik%2], langs[(ik+1)%2])
                            #print("len(back_src_sents), len(nmted_sents) : {}, {}".format(len(back_src_sents), len(nmted_sents)))
                            #train_data_back = list(zip(nmted_sents, back_src_sents)) 
                            #train_data_back = list(zip(list(zip(nmted_sents, back_src_sents[:len(nmted_sents)])),sent_scores))
                            temp_set = list(zip([s.split(' ') for s in nmted_sents], back_src_sents, sent_scores))
                            # 번역된 문장의 길이가 원문에 비해 너무 짧지 않고 중복되는 단어가 적고(문장길이/set길이) 스코어가 높은 번역 
                            train_data_back = []
                            for bi in temp_set:
                                b0 = [w for w in bi[0] if w not in sbol]
                                if len(bi[0])/len(bi[1]) > lenR:
                                    train_data_back.append([bi[0],bi[1],1./len(b0)+len(b0)/len(set(b0))-bi[2]])

                            #print("translated, src_sent, score : {} {} {}".format(
                            #    train_data_back[10][0].split(' ')[:20],train_data_back[10][1][:20],train_data_back[10][2]))
                            #print("len(train_data_back) : {} {}".format(len(train_data_back), train_data_back[-1])) 
                            cutline = sorted([bi[2] for bi in train_data_back])[(valid_niter * (train_batch_size//(k_div*b_ratio) + batch_plus))*1]
                            cutlines.append(cutline)
                            #train_data_back = [[bi[0],bi[1]] for bi in train_data_back if bi[2] < cutline] 
                            #print(train_data_back[:2])
                            biX = [bi[0] for bi in train_data_back if bi[2] < cutline] 
                            biY = [bi[1] for bi in train_data_back if bi[2] < cutline] 

                            print(f"len(biX) : {len(biX)}, len(biY) : {len(biY)}")
                            
                            with open('en_es_data/pseudo_bi_src/en_src' if ik==0 else 'en_es_data/pseudo_bi_src/ko_src','a') as f: # if inK>0 else 'w') as f:
                                f.write('\n'+'\n'.join([' '.join(s) for s in biY]))
                            with open('en_es_data/pseudo_bi_src/ko_made' if ik==0 else 'en_es_data/pseudo_bi_src/en_made','a') as f: # if inK>0 else 'w') as f:
                                f.write('\n'+'\n'.join([' '.join(s) for s in biX]))

                    print("load back_translated data to make batch generator")
                    for ik,sl in enumerate(['ko2en', 'en2ko']):
                        with open('en_es_data/pseudo_bi_src/en_src' if ik==0 else 'en_es_data/pseudo_bi_src/ko_src','r') as f:
                            biY = [s.split(' ') for s in f.read().split('\n')[-nK*valid_niter*train_batch_size:]]
                        with open('en_es_data/pseudo_bi_src/ko_made' if ik==0 else 'en_es_data/pseudo_bi_src/en_made','r') as f:
                            biX = [s.split(' ') for s in f.read().split('\n')[-nK*valid_niter*train_batch_size:]]

                        print(f"len(biX) : {len(biX)}, len(biY) : {len(biY)}")
                        
                        if back_trans_X==1:
                            #train_data_back_b = [[bi[0],bi[1]] for bi in train_data_back if bi[2] < cutline] 
                            
                            train_data_back_b = list(zip(biX,[['_','<s>'] + sent + ['_','</s>'] for sent in biY]))
                            # back_trans 는 당분간 normal 과 2:1로 결합해서 ...즉 1/2 씩 적용 
                            mK = 5  #1 if self_trans_X ==1 else 5  #if back trans only, mK = 1, elif mix with normal then mK=5
                            b_trns[sl] = batch_iter(train_data_back_b, train_batch_size//(mK*k_div*b_ratio) + batch_plus, langs[(ik+1)%2], langs[ik%2], shuffle=True) 
                            bt_batch = mK * nK
                        
                        if self_trans_X==1:
                            #train_data_back_s = [[bi[1],bi[0]] for bi in train_data_back if bi[2] < cutline]
                             
                            train_data_back_s = list(zip(biY,[['_','<s>'] + sent + ['_','</s>'] for sent in biX]))
                            s_trns[sl2[ik]] = batch_iter(train_data_back_s, train_batch_size//(k_div*b_ratio) + batch_plus, langs[ik%2], langs[(ik+1)%2], shuffle=True, self_trns=1) 
                            st_batch = 1 * nK
                        """
                        with open('en_es_data/pseudo_bi_src/en_src' if ik==0 else 'en_es_data/pseudo_bi_src/ko_src','a') as f:
                            f.write('\n'.join([' '.join(s) for s in biY]))
                        with open('en_es_data/pseudo_bi_src/ko_made' if ik==0 else 'en_es_data/pseudo_bi_src/en_made','a') as f:
                            f.write('\n'.join([' '.join(s) for s in biX]))
                        """
                    cut_lines= [sum([c for i,c in enumerate(cutlines) if i%2==0])/nK, sum([c for i,c in enumerate(cutlines) if i%2==1])/nK]
                    print("{} sents are back-translated .......    cutlines : {}  {}".format(len(nmted_sents)*2*nK,cut_lines[0], cut_lines[1]))

            #optimizer.zero_grad()

            batch_size = train_batch_size #len(src_sents)

            batch_loss = 0
            loss_base = 0
            #langs = [slang, tlang]

            ####################
            #if True: #mapping == 0:
            if mapping == 0 and self_learning == 0 and back_trans==0 and self_trans==0:
                #src_sents, tgt_sents = next(forward_batch)
                example_losses1 = -model(src_sents, tgt_sents, slang, tlang) # (batch_size,)           
                src_sents2, tgt_sents2 = next(reverse_batch)  
                example_losses2 = -model(src_sents2, tgt_sents2, tlang, slang) # (batch_size,)
                batch_loss += example_losses1.sum() + example_losses2.sum()
                loss_base += 2
            #####################
            if self_learning==1:
                src_sents3, tgt_sents3 = next(eng_batch)  
                example_losses3 = -model(src_sents3, tgt_sents3, slang, slang) # (batch_size,)
                src_sents4, tgt_sents4 = next(kor_batch)  
                example_losses4 = -model(src_sents4, tgt_sents4, tlang, tlang) # (batch_size,)
                batch_loss += (example_losses3.sum() + example_losses4.sum()) * (1.0 / l_ratio)
                loss_base += 2 / (b_ratio * l_ratio)

            if back_trans==1:
                if train_iter % valid_niter ==1:
                    print(f"start back_training!  bt_batch = {bt_batch}")
                    bt_batch -= 1
                    
                """
                
                for ik, lkey in enumerate(['en2ko','ko2en']):                   
                    if ik==1:
                        src_sents, tgt_sents = next(reverse_batch)
                    src_sents3, tgt_sents3 = next(b_trns[lkey]) 
                    src_sents = src_sents3
                    tgt_sents = src_sents3
                    #src_sents, tgt_sents = sents_ordering(list(zip(src_sents,tgt_sents)))
                    if ik == 0:
                        batch_loss += -model(src_sents, tgt_sents, slang, tlang).sum()
                    else:
                        batch_loss += -model(src_sents, tgt_sents, tlang, slang).sum()

                    #batch_loss += -model(src_sents, tgt_sents, langs[ik%2], langs[(ik+1)%2]).sum()
                loss_base += 2

                """
                src_sents2, tgt_sents2 = next(reverse_batch)

                src_sents_bt3, tgt_sents_bt3 = next(b_trns['en2ko'])
                #print('\n',src_sents3[7]) 
                #print(tgt_sents3[7],'\n')
                if self_trans ==1: #self_trans_X ==1: # 
                    example_losses3 = -model(src_sents_bt3, tgt_sents_bt3, slang, tlang, back_trns =1) 
                else:
                    src_sents, tgt_sents = sents_ordering(list(zip(src_sents+src_sents_bt3,tgt_sents+tgt_sents_bt3)))
                    example_losses3 = -model(src_sents, tgt_sents, slang, tlang) # (batch_size,)

                src_sents_bt4, tgt_sents_bt4 = next(b_trns['ko2en']) 
                if self_trans ==1: #self_trans_X ==1: #   # 설명 : self_trans_X ==1 로 할 경우의 설명 : 현재 iter에서 self_trans 가 1 이 아니더라도 self_trans_X 가 1이면 단독으로 (메모리 문제도 고려)
                    example_losses4 = -model(src_sents_bt4, tgt_sents_bt4, slang, tlang, back_trns =1)  
                else:              
                    src_sents2, tgt_sents2 = sents_ordering(list(zip(src_sents2+src_sents_bt4,tgt_sents2+tgt_sents_bt4))) 
                    example_losses4 = -model(src_sents2, tgt_sents2, tlang, slang) # (batch_size,)

                batch_loss += (example_losses3.sum() + example_losses4.sum()) * (1.0 / l_ratio)
                loss_base += 2 #/ (b_ratio * l_ratio) 


                """
                src_sents3, tgt_sents3 = next(b_trns['en2ko'])
                #print('\n',src_sents3[7]) 
                #print(tgt_sents3[7],'\n')
                example_losses3 = -model(src_sents3, tgt_sents3, slang, tlang) # (batch_size,)
                src_sents4, tgt_sents4 = next(b_trns['ko2en'])  
                example_losses4 = -model(src_sents4, tgt_sents4, tlang, slang) # (batch_size,)
                batch_loss += (example_losses3.sum() + example_losses4.sum()) * (1.0 / l_ratio)
                loss_base += 2 / (b_ratio * l_ratio) 
                """   

            if self_trans==1:  
                if train_iter % valid_niter ==1:
                    print(f"start self_training!  st_batch = {st_batch}")
                    st_batch -= 1
                    

                """# 잠정 테스트로 ..................................추가
                if train_iter % (valid_niter * 8) ==1:
                    l_rate *= 0.5
                    print(f"learning rate is lowered to {l_rate}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = l_rate  
                """# ...............................................                  

                #src_sents3, tgt_sents3 = next(b_trns['en2ko'])
                src_sents_st3, tgt_sents_st3 = next(s_trns['en2ko'])  # ko: translated, en: src  
                #print(f"src_sents_st3 : {src_sents_st3[0]}")  
                #print(f"tgt_sents_st3 : {tgt_sents_st3[0]}")            
          

                #src_sents, tgt_sents = sents_ordering(list(zip(src_sents3,tgt_sents3)))
                example_losses_st3 = -model(src_sents_st3, tgt_sents_st3, slang, tlang, self_trns=1) # (batch_size,)

                src_sents_st4, tgt_sents_st4 = next(s_trns['ko2en']) 
                #print(f"src_sents_st4 : {src_sents_st4[0]}")  
                #print(f"tgt_sents_st4 : {tgt_sents_st4[0]}")  
                #src_sents2, tgt_sents2 = sents_ordering(list(zip(src_sents4,tgt_sents4))) 
                example_losses_st4 = -model(src_sents_st4, tgt_sents_st4, tlang, slang, self_trns=1) # (batch_size,)

                batch_loss += (example_losses_st3.sum() + example_losses_st4.sum()) * (1.0 / l_ratio)
                loss_base += 2//k_div #/ (b_ratio * l_ratio) 


            if mapping==1:   
                if train_iter == 1 or train_iter % 100 == 1:
                    trained_W = [model.map_en, model.map_ko]
                    nul_mapping = True if train_iter==1 else False
                    update_map = False

                    if train_iter == 1:
                        update_map = True
                       
                    elif map_learning==1:
                        m_diff = torch.norm(torch.mm(trained_W[0].weight.data,trained_W[1].weight.data) - torch.eye(300).to(device))
                        if m_diff > 1.0:   
                            update_map = True                     
                            print("torch.mm(W[0],W[1] is not Identity Matrix. difference = {}".format(m_diff))
                    elif train_iter % 200 == 1:
                        update_map = True
                    
                    if update_map==1:
                        with torch.no_grad():
                            map_en, map_ko,dict_mapping = net_run(model.model_embeddings, model.vocab, trained_W, nul_mapping, device) 
                        model.map_en.weight.data = map_en if train_iter == 1 else 0.9 * map_en + 0.1 * model.map_en.weight.data
                        model.map_ko.weight.data = map_ko if train_iter == 1 else 0.9 * map_ko + 0.1 * model.map_ko.weight.data 
                        model.dict_mapping = dict_mapping

                src_sents5, tgt_sents5 = next(eng_mapping_batch)  
                example_losses5 = -model(src_sents5, tgt_sents5, slang, slang, mapping=1) # (batch_size,)
                src_sents6, tgt_sents6 = next(kor_mapping_batch)  
                example_losses6 = -model(src_sents6, tgt_sents6, tlang, tlang, mapping=1) # (batch_size,)
                batch_loss += (example_losses5.sum() + example_losses6.sum()) * (1.0 / l_ratio)
                loss_base += 2 / (b_ratio * l_ratio)      
            #example_losses = -model(src_sents, tgt_sents, slang, tlang) # (batch_size,)
            #batch_loss = example_losses.sum() 
            batch_loss = batch_loss / loss_base
            loss = batch_loss / batch_size
            
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                #print("learning rate : {}".format(lr))

                # compute dev. ppl and bleu, dev batch size can be a bit larger
                dev_ppl = evaluate_ppl(model, dev_data, slang, tlang, batch_size=64)   
                valid_metric = -dev_ppl
                cut_line += c_plus

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores) - (patience + 1) * (0.00002 +self_trans*0.2)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')

                    n_save += 1
                    dn = "../drive/My Drive/temp2" if n_save%2 ==0 else "../drive/My Drive/temp3"
                    shutil.copy2("model.bin.optim",dn)
                    shutil.copy2("model.bin",dn)
                    shutil.copy2("net_Module/dict_en.json", dn)
                    shutil.copy2("net_Module/dict_ko.json", dn)

                    print("current model was saved !!!")
                elif patience < patience_ceiling:  #int(args['--patience']):
                    patience += 1
                    
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == patience_ceiling: #int(args['--patience']):
                        #patience_ceiling += 12
                        num_trial += 1
                        #print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
                """
                if mapping:
                    with torch.no_grad():
                        map_en, map_ko, dict_mapping = net_run(model.model_embeddings, model.vocab, device)
                    model.map_en.weight.data = 0.4 * map_en + 0.6 * model.map_en.weight.data
                    model.map_ko.weight.data = 0.4 * map_ko + 0.6 * model.map_ko.weight.data
                    model.dict_mapping = dict_mapping
                """
                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    wid2cid = get_wid2cid()
    char_size = 85
    model = NMT.load(args['MODEL_PATH'], char_size=char_size, wid2cid=wid2cid)

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             tlang = args['--tlang'])

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

def beam_search(model, test_data_src, beam_size, max_decoding_time_step, tlang): 
    """
    def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) 
    -> List[List[Hypothesis]]:
    Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    print(tlang, "tlang == ko : {}".format(tlang == 'ko' ))

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.ek_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step,tlang=tlang)
            """
            if tlang =='en':
                example_hyps = model.ke_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            else:
                example_hyps = model.ek_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            """
            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def decode_greedy(args: Dict[str, str]):
 
    print("load sentences for back_Trans from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)

    wid2cid = get_wid2cid()
    char_size = 85
    model = NMT.load(args['MODEL_PATH'], char_size=char_size, wid2cid=wid2cid)
    #model = NMT.load(args['MODEL_PATH'])
    

    mapping = args['--mapping']
    #print(mapping)
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    model.to(device)

    if mapping==1:   
        with torch.no_grad():
            map_en, map_ko,dict_mapping = net_run(model.model_embeddings, model.vocab, device) 
        model.map_en.weight.data = map_en 
        model.map_ko.weight.data = map_ko 
        model.dict_mapping = dict_mapping

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    nmted_sents, sent_scores = greedy_search(model, test_data_src,
                             batch_size=int(args['--batch-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             slang = args['--slang'],
                             tlang = args['--tlang'],
                             mapping = 0) #mapping)
    """
    if args['TEST_TARGET_FILE']:
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, nmted_sents)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)
    """
    to_add = '_mapping' if mapping==1 else ''
    print("length of nmted : {}".format(len(nmted_sents)))

    with open(args['OUTPUT_FILE']+to_add, 'w') as f:
        for src_sent, nmted, score in list(zip(test_data_src[:len(nmted_sents)], nmted_sents, sent_scores)):
            sent = ''.join(nmted)
            src = ' '.join(src_sent)
            #with open('outputs/bcktr_en2ko.txt', 'w' if i == 1 else 'a') as f:
            #f.write(str(score) + '\n' + src +'\n' + sent + '\n' +'\n')
            f.write(sent + '\n')

def greedy_search(model, test_data_src, batch_size, max_decoding_time_step, slang, tlang, mapping): 

    was_training = model.training
    model.eval()

    #print(tlang, "tlang == ko : {}, mapping : {}".format(tlang == 'ko', mapping ))

    nmted_sents = []
    sents_scores = []
    n_iter = len(test_data_src) // batch_size + 1
    with torch.no_grad():
        #for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        for ik in range(n_iter):
            src_snts = test_data_src[ik*batch_size:(ik+1)*batch_size]
            if len(src_snts) <2:continue
            sents, scores = model.greedy_search(src_snts, max_decoding_time_step=max_decoding_time_step,slang=slang,tlang=tlang,mapping=mapping)
            #print("length of sents : {}, sents[0][:3] : {}".format(len(sents), sents[0][:3]))
            nmted_sents += sents 
            sents_scores += scores
        #print("length of nmted_sents : {}, sents[0][:3] : {}".format(len(nmted_sents), nmted_sents[10][:100]))
        #print(src_snts[10][:100])
    if was_training: model.train(was_training)

    return nmted_sents, sents_scores


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    """
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    """

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    elif args['backtr']:
        decode_greedy(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
