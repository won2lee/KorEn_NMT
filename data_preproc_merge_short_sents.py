# 짧은 문장 합치기
from itertools import chain

with open('fromTED/train6.en', 'r') as f:
    en = f.read().split('\n')
with open('fromTED/train6.ko', 'r') as f:
    ko = f.read().split('\n')

p = re.compile('thank\s+\_\s+you')
q = re.compile('\_\s\(\s\^\sapplause.*') 
startX = list(chain(*[[i,i+1] for i,s in enumerate(en[:-2]) 
          if p.search(s) is not None and q.search(en[i+1]) is not None]))

shortS = [i for i,s in enumerate(en[:-2]) 
          if len(s) <50 and len(en[i+1])<150 and i not in startX ]
shortS = [s for i,s in enumerate(shortS[1:]) if s != shortS[i]+1]
shortSX = [s+1 for s in shortS]

enM = [s if i not in shortS else s+' '+en[i+1] for i,s in enumerate(en)]
enM = [s for i,s in enumerate(enM) if i not in shortSX]

koM = [s if i not in shortS else s+' '+ko[i+1] for i,s in enumerate(ko)]
koM = [s for i,s in enumerate(koM) if i not in shortSX]

with open('train6.en','w') as f:
    f.write('\n'.join(enM))
with open('train6.ko','w') as f:
    f.write('\n'.join(koM))
len(enM), len(koM)