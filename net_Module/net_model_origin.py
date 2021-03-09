import os
import math
import random
import numpy as np
import scipy
import scipy.linalg
import torch
import torch.nn as nn
from net_Module.net_util import get_candidates, build_dictionary, dict_merge
#import torch.nn.functional as F

#from build_dict import get_candidates, build_dictionary

class Net(nn.Module): #Word Mapping

    def __init__(self, embedding, vocab, device, dropout_rate=0.2):
        super(Net, self).__init__()
        self.hidden_size = 300
        self.ko_start = 54621
        #self.dropout_rate = dropout_rate
        #self.vocab = vocab
        #self.token = ['(', ')', ',', "'", '"','_','<s>','</s>']
        #self.sbol = ['_','^','`']
        self.top_k = 10
        self.some_number = 0.3
        self.best_valid_metric = 0
        self.vocs = vocab.vocs #parameters['vocab'].vocs
        self.emb = embedding.vocabs.weight.data #embedparameters['state_dict']['model_embeddings.vocabs.weight'].to("cuda:0") #.cpu() #to("cuda:0")
        self.device = device
        #self.lookup = {}
        
        self.mapping = None
        self.mapping = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        #self.map = [torch.zeros([self.hidden_size, self.hidden_size], device=device) for i in range(2)]
        

    def forward(self, dictionary, slang): 
             
        #if type(dictionary[0][0]) is str:
        dictionary = torch.tensor([(self.vocs[b[0]], self.vocs[b[1]]) for b in dictionary])
        #print("dictionary.size : {}".format(dictionary.size()))
        W = self.procrustes(dictionary)
        self.mapping.weight.data = W
        mean_cosine, dico, scores = self.dist_mean_cosine(slang)
        #self.save_best(mean_cosine, slang)
        """
        lk = 0 if slang == 'en' else 1
        if self.map[lk][0,0] == 0.:
            self.map[lk] = W
        else:
            self.mapping.weight.data = 0.1 * W + 0.9 * self.map[lk]
            self.map[lk] = self.mapping.weight.data
        """
        return self.mapping.weight.data, mean_cosine, dico, scores
           
    
    def procrustes(self, dico):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.emb[dico[:, 0]]         #torch.tensor(dico[:, 0])]
        B = self.emb[dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W = torch.tensor(U.dot(V_t)).to(self.device) #.cpu()   #.copy_(torch.from_numpy(U.dot(V_t)).type_as(W)) 
        #print("W.size : {}".format(W.size()))
        return W
        
    def dist_mean_cosine(self, slang):
        """
        Mean-cosine model selection criterion.
        """
        n_wds = 35000
        # get normalized embeddings
        #src_emb = self.mapping(self.emb[:self.ko_start])
        #tgt_emb = self.emb[self.ko_start:]
        if slang == 'en':
            src_emb = self.mapping(self.emb[:n_wds])
            tgt_emb = self.emb[self.ko_start:self.ko_start+n_wds]
        else:
            src_emb = self.mapping(self.emb[self.ko_start:self.ko_start+n_wds])
            tgt_emb = self.emb[:n_wds]           
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True) #.expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True) #.expand_as(tgt_emb)

        # build dictionary
        for dico_method in ['csls_knn_10']:
            dico_build = 'S2T'
            dico_max_size = 50000
            # temp params / dictionary generation
            _params = {}

            #_params = deepcopy(self.params)
            _params['dico_method'] = dico_method # 'csls_knn_10'
            _params['dico_build'] = dico_build
            _params['dico_threshold'] = 0
            _params['dico_max_rank'] = 50000
            _params['dico_min_size'] = 0
            _params['dico_max_size'] = dico_max_size
            
            s2t_candidates = get_candidates(src_emb, tgt_emb, _params, self.device)
            #t2s_candidates = get_candidates(tgt_emb, src_emb, _params, self.device)
            dico, scores = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates ) #, t2s_candidates)
            # mean cosine
            if dico is None:
                print("dico is None !!!!!")
                mean_cosine = -1e9
            else:
                #print(dico.size())
                dico_max_size = 20000
                mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                #print("mean_cosine of {} pairs = {}".format(dico_max_size, mean_cosine))
            #mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
            print("Mean cosine (%s method, %s build, %i max size): %.5f"
                        % (dico_method, _params['dico_build'], dico_max_size, mean_cosine))
            #to_log['mean_cosine-%s-%s-%i' % (dico_method, _params['dico_build'], dico_max_size)] = mean_cosine
            
        return mean_cosine, dico, scores
        
    def save_best(self, metric,slang):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if metric > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = metric
            #logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data #.cpu().numpy()
            path = os.path.join('net_Module/', 'best_mapping_en.pth' if slang=='en' else 'best_mapping_ko.pth')
            print('* Saving the mapping to %s ...' % path)
            torch.save(W, path)
    
    
    def make_lookup(self, src, tgt):
        for ws in src:
            scores, w_topk = torch.topk(cos_sim(self.w_map(self.emb[wD[ws]]).expand(batch,self.hidden_size),
                              embed[self.ws2inds(tgt)]), self.top_k)
            wt = w_topk[0]
            scores_sk, ws_topk = torch.topk(cos_sim(self.emb[wD[ws]].expand(batch,self.hidden_size),
                              embed[self.ws2inds(src)]), self.top_k)
            scores_tk, wt_topk = torch.topk(cos_sim(self.emb[wt].expand(batch,self.hidden_size),
                              embed[self.ws2inds(tgt)]), self.top_k)
            CSIL = 2* cos_sim(self.w_map(self.emb[wD[ws]]),self.emb[wt]) - (sum(scores_sk)+ sum(scores_tk)) / self.top_k
            if CSIL > self.some_number:
                self.lookup[ws].append((wt,CSIL))   
        return self.lookup
            
    def ws2inds(self, words):  
        return [self.vocs[w] for w in words]
    
    def cos_sim(self,a,b):
        return sum(a*b)/((sum(a*a)**.5)*(sum(b*b)**.5))

    
def batch_iter(bi_words,b_size):
    batch_size = b_size
    batch_num = math.ceil(len(bi_words) / batch_size)
    index_array = list(range(len(bi_words)))

    np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [bi_words[idx] for idx in indices]
        src = [e[0] for e in examples]
        tgt = [e[1] for e in examples]

        yield src, tgt
