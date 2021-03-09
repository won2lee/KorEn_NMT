
import re
import collections
from collections import Counter
from itertools import chain
from tqdm.notebook import tqdm

inP = "../drive/My Drive/en_es_data/train"
"""
def count_from_file(file_list):
    c = Counter()
    for f in tqdm(file_list):
        d_out =  get_text(f)
        c = c + Counter(d_out.split(' '))
    return c
"""
p = re.compile('(?P<sbol>[\_\^\`])')
c = Counter()
for i in tqdm(range(1,8)):
    with open(inP+str(i)+".en",'r') as f:
        X = f.read().split('\n')
    c = c+Counter(list(chain(*[[w for w in p.sub('Æ\g<sbol>',s).split('Æ') if len(w)>0 and w[0] in ['^','`']] for s in X])))

for i in tqdm(range(1,3)):
    with open("../drive/My Drive/en_es_data/totrain"+str(i)+".en",'r') as f:
        X = f.read().split('\n')
    c = c+Counter(list(chain(*[[w for w in p.sub('Æ\g<sbol>',s).split('Æ') if len(w)>0 and w[0] in ['^','`']] for s in X])))

dictC = dict(c)
print(len(dictC))

listD = sorted(dictC.items(), key=lambda x:x[1], reverse=True)
listD[45000:45100]

XX = [w[0] for w in listD[1000:50000]]
import numpy as np
import random
XXX = []
for i in range(30000):
    k = np.random.randint(15,40)
    klist = random.sample(range(len(XX)), k)
    s = '_ . '.join([XX[i] for i in klist])
    XXX.append(s)

with open("../drive/My Drive/en_es_data_test/train8.en",'w') as f:
    f.write('\n'.join(XXX))

import json
with open('en_es_data/e2e_train', 'w') as f:
    json.dump(dictC,f)