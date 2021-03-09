nR = 0
deN = 3
with open('train_CNN/new_data/en2ko/srcEn','r') as f:
    en = f.read().split('\n')
en = [s for i,s in enumerate(en) if i%deN == nR]
with open('en_es_data/srcE', 'w') as f:
    f.write('\n'.join(en))
    
with open('train_CNN/new_data/to_Train_0515/srcKo','r') as f:
    ko = f.read().split('\n')
ko = [s for i,s in enumerate(ko) if i%deN == nR]
with open('en_es_data/srcK', 'w') as f:
    f.write('\n'.join(ko))
            
with open('en_es_data/train_donot_change/train.en','r') as f:
    en = f.read().split('\n')
en = [s for i,s in enumerate(en) if i%deN == nR]
with open('en_es_data/train.en', 'w') as f:
    f.write('\n'.join(en))
            
with open('en_es_data/train_donot_change/train.ko','r') as f:
    ko = f.read().split('\n')
ko = [s for i,s in enumerate(ko) if i%deN == nR]
with open('en_es_data/train.ko', 'w') as f:
    f.write('\n'.join(ko))
