
import torch
import torch.nn as nn

class CharEmbeddings(nn.Module): 

    def __init__(self, char_size, embed_size, hidden_size):
        super(CharEmbeddings, self).__init__()
        self.char_embeddings = nn.Embedding(char_size, 30)
        self.projection = nn.Linear(embed_size, hidden_size, bias=False)
    
    def forward(self, X):
        X = self.projection(self.char_embeddings(torch.tensor(X)).view(-1,embed_size).contiguous)
        return X
