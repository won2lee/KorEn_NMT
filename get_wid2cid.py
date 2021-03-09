
import torch
import json
import torch.nn as nn
#import torch.nn.utils
from vocab import Vocab, VocabEntry

def gen_wid2cid():
    char_list = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|
    _@#$%^&*~`+-=<>()[]""")
    char_list += ['”','“','—','’','˅','‘','é','…','–','□','•','á','·','▪','í','ʹ','●','ó','ü','◆','°','´']
    char_list += ['㎡']
    char2id = dict() # Converts characters to integers
    char2id['<pad>'] = 0

    for i, c in enumerate(char_list):
        char2id[c] = len(char2id)

    char_size = len(char2id)

    vocab = Vocab.load('vocab.json')

    wid2cid = {}
    for i in range(len(vocab.vocs)):
        w = [char2id[c] if c in char2id.keys() else 0 for c in vocab.vocs.id2word[i]]
        wid2cid[i] = w[:10] + [0]*(10-len(w))

    json.dump(wid2cid, open('wid2cid.json', 'w'), indent=2)

###########################################
class CharEmbeddings(nn.Module): 

    def __init__(self, char_size):

        super(ModelEmbeddings, self).__init__()
        self.char_embeddings = nn.Embedding(char_size, 30)
        self.projection = nn.Linear(300, 300, bias=False)
    
    def forward(self, X):
        X = self.projection(self.char_embedding(torch.tensor(X)).view(-1,300).contiguous)
        return X

upper_index = torch.tensor((mask == 1))
upperX = [[wid2cid[str(w)] for w in ws] for ws in batch[upper_index]]
self.char_emb = CharEmbeddings(char_size)
upperX = self.char_emb(upperX)
X = torch.zeros(batch_size, emb_size).float().to(device)
X[upper_index] = upperX
