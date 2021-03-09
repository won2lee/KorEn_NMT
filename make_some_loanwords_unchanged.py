
import numpy as np
import copy
import re
from itertools import chain
from tqdm.notebook import tqdm

def make_nm_dict(in_f,out_f):
    lang = ['en','ko']
    fns = []
    for i,l in enumerate(lang):
        with open(in_f+l,'r') as f:
            fns.append(f.read().split('\n'))
    for ik,s in enumerate(fns[0]): 
        if len(s.split(','))!=len(fns[1][ik].split(',')):
            print(s,'\n',fns[1][ik])

    fns = [[s.split(',') for s in fn] for fn in fns]


    bi_loans = [[(fns[0][ik][il],nm) for il,nm in enumerate(s) if nm[0] not in ['^','`']] 
               for ik,s in enumerate(fns[1])] 
    Xen = [','.join([p[0] for p in ps]) for ps in bi_loans]
    Xko = [','.join([p[1] for p in ps]) for ps in bi_loans]

    X =[Xen,Xko]
    for i,l in enumerate(lang):
        with open(out_f+l,'w') as f:
            f.write('\n'.join(X[i]))

    bi_list = list(chain(*bi_loans))
    nm_list = list(set(bi_list))

    p = re.compile('(?P<sbol>[\_\^\`])')
    nm_temp = [(p.sub('Æ \g<sbol>',n[0]),p.sub('Æ \g<sbol>',n[1])) for n in nm_list]

    sn_list = list(chain(*[list(zip(n[0].split('Æ '), n[1].split('Æ '))) 
                        for n in nm_temp if len(n[0].split('Æ ')) == len(n[1].split('Æ '))]))
    sn_list2 = list(set(sn_list))
    sn_list2[:100], len(sn_list2)

    nm_dict = dict(sn_list2)
    k2e_dict = {v:k for k,v in nm_dict.items()}
    
    return nm_dict, k2e_dict


def load_data(in_f):
    lang = ['en','ko']
    zz = re.compile('\_\s*\_')
    
    dns2 = []
    for i,l in enumerate(lang):
        with open(in_f+l,'r') as f: #train4_before.  train1.
            dns2.append(f.read().split('\n'))
    dns2 = [[zz.sub('_',s) for s in ld] for ld in dns2] 
    
    return dns2


def re_sub(x):
    z = re.compile('[a-z0-9가-힣]')
    xx = []
    for c in x:
        if z.match(c) is not None:
            xx.append(c)
        elif c==' ':
            xx.append('\s')
        else:
            xx.append('\\'+c)
    return ''.join(xx)


def inject_e2e_dataf(dns):

    n1 =n2 =n3=0
    p = re.compile('(?P<sbol>[\_\^\`])')
    q = re.compile(',')
    z = re.compile('\s+')

    for i,s in enumerate(tqdm(dns[0])):
        soN = 0
        k2e = 0
        pre_nm_en = []
        pre_nm_ko = []

        for nm in p.sub('Æ \g<sbol>',s).split('Æ '):       
            nm = q.sub('',nm).strip()
            if nm == '': continue

            if nm[0] in ['^','`']:
                soN += 1  #start of name
                n1+=1
                if  nm in nm_dict.keys() and nm_dict[nm] in dns[1][i]:
                    n2+=1

                    if k2e==1 or np.random.randint(1,6)<len(nm_dict[nm].strip().split(' ')):
                        n3+=1                    

                        dns[1][i] = re.sub(re_sub(nm_dict[nm]),nm,dns[1][i])

                        if n3<10:
                            print(nm, nm_dict[nm], re_sub(nm_dict[nm]))
                            print(dns[1][i])

                        if soN>1 and k2e == 0 and len(pre_nm_en)>0:
                            print("posterior change !!!!!!!!!!")
                            print(dns[1][i])
                            for i_en in range(len(pre_nm_en)):
                                dns[1][i] = z.sub(' ',re.sub(re_sub(pre_nm_ko[i_en]),pre_nm_en[i_en]+' ',dns[1][i]))
                            print(dns[1][i])

                        k2e = 1 #ko to en

                    pre_nm_en.append(nm)
                    pre_nm_ko.append(nm_dict[nm])

                    #else:
                        #k2e = 0
                        #pre_nm_en = ''
                else:
                    k2e=0 
                    pre_nm_en = []
                    pre_nm_ko = []

            else:
                soN = 0
                k2e = 0
                pre_nm_en = []
                pre_nm_ko = []
                  
    
    print(n1,n2,n3)
    return dns

""" 
import make_some_loanwords_unchanged

nm_dict,k2e_dict = make_nm_dict('data/train4_before.','train4_before.')
dns2 = load_data('data/train1.')
dns = copy.deepcopy(dns2)
dns = inject_e2e_dataf(dns)

nn = 0            
for i,s in enumerate(dns[1]):
    if s!=dns2[1][i]:
        nn+=1
        if nn<10:
            print(s,'\n',dns2[1][i])
    
"""    
