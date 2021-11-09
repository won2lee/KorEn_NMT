# Korean-English Translation Model

##### 본 모델 개요도 

<img src="/images/KE_NMT.jpg" width="700px" title="본 모델 개요도" alt="KE_NMT"></img><br/>



#### 데이터: 
- 한글영어 번역 데이터 :  AI hub (110만개), OPUS의 JW300, ELRC, Ted 번역 데이터
- 모노코퍼스 : 한글 영어 위키피디아, 영어 뉴스 등 

#### 한글/영어 선 처리  :           
     한글의 경우 아래 예와 같이 어간/어미, 체언/조사 등이 필요에 따라 ㄱ,ㄴ 등 한글 알파벳 수준까지 분리되어 처리되며      
     영어의 경우 단복수 시제 합성어 대소문자 등을 고려하여 단어를 분리 처리

(한글 선처리 예)
- 심도있는 논의와 검토를 거쳤다 > 심도 있 는 _ 논의 와  _ 검토 를 _ 거치 ㅓㅆ 다 
- 책임을 덮어씌웠다                    > 책임 을 _ 덮 어 씌우 ㅓㅆ 다 
                   
#### 모델 구성: 
    일반적인 Encoder/Decoder(LSTM 2 Layer)를 근간으로 하지만 Sub Layer 와 Mapping Module 이 추가 

- Sub Layer : Embed Vecor 를 Encode/Decode LSTM 으로 보내기전 어절 단위로 묶어 처리하는 Layer
- Mapping Module : 한글과 영어의 Mapping matrix를 update (Mapping training 할 경우)

#### 학습(Training):
    번역문이 있는 문장(원문 번역문) 들에 대한 일반적인 학습 뿐 아니라     
    번역문이 없는 Mono Corpus (위키피디아 뉴스 등)에 대해서     
    Self Training, Back-translation, 'Mapping training' 등 병행     
    (이 과정을 통하여 dev set의 perplexity를 7-8% 정도 추가로 낮춤)   

- Self Training : 주로 Encoder 부분 학습 
- Back-translation : 주로 Decoder 부분 학습
- Mapping training: 영어와 한글의 embedding vector를 상호 align

  CSLS 스코어가 높은 단어짝을 기준으로 한영 단어간 Mapping Matrix 를 만들고     
  이를 모델에 주입해 한영 Embed Vector를 align 시키는 Mapping  training 을 통해      
  한글-영어 단어짝의 cosine similarity (상위 20000개 단어짝의 평균)를 0.32 수준에서 0.46 수준으로 끌어올림.     
  CSLS (Cross-domain similarity local scaling)의 개념은 
  “Word Translation Without Parallel Data” (Conneau et al., 2017)에서 원용, aligning 방법은 직접 개발

#### 번역(Beam Search):  
- length normalization + coverage penalty (attention matrix 이용) 반영 

#### 모델 디플로이: 
   영어는 한글로, 한글은 영어로 번역    
   https://github.com/won2lee/KE_Translator (python, pytorch, Flask 기반)    
   https://fluent-outpost-276916.uc.r.appspot.com/ (set disabled now)   


