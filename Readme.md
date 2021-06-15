# 텍스트 데이터를 활용한 MBTI 분석

### 기본 정보 
+ 소속/이름/학번 : 산업정보시스템 16102166 김봉석

+ 프로젝트명: 텍스트 데이터를 활용한 MBTI 성격을 분류, 특징 관찰!  

+ 팀 구성 : 16102166 산업정보시스템 김봉석(1인)

### 계획
 + 데이터 크롤링
 + 수집한 데이터를 활용한 MBTI별 Wordcloud
 + ML을 활요한 MBTI 분류
 + Kobert를 활용한 MBTI 분류

### 프로젝트 소개

### Example Sample image (수집한 전체데이터에 대한 word cloud 결과)
<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/total_personlity.png"  width="40%">


작품소개:  텍스트 데이터를 가지고 MBTI 성격을 분류 & 특징 관찰을 해보자 ! 
최근까지 유행을 하고 있는 MBTI 16가지 성격으로 구분되는 검사를, 사람의 텍스트 데이터만으로 성격을 예측할 수 있을까? , 성격에 따른 텍스트의 특징은 무엇이 있을까?


#### 왜 만들고자 하는가? (재미 + 유용성 타입)

#### 1) 재미 
+ MBTI 성격유형 검사 란?
마이어스-브릭스 유형 지표는 개인이 쉽게 응답할 수 있는 자기보고서 문항을 통해 인식하고 판단할 때의 각자 선호하는 경향을 찾고, 이러한 선호 경향들이 인간의 행동에 어떠한 영향을 미치는가를 파악하여 실생활에 응용할 수 있도록 제작된 심리 검사이다(위키백과)
+ 사람의 성격은 사람의 필체나, 문장, 말투를 통해서 표출이 된다고 하는데, 그렇다면 MBTI 또한 사람의 텍스트 데이터를 이용해서 성격을 분류하고 특징을 뽑아 낼 수 있지 않을까? 라는 단순한 생각에서 시작되었습니다!

#### 2) 유용성 
+ 흔히 쉽게 인터넷에서 할 수 있는 짧은 MBTI 검사입니다. 그럼에도 불구하고 수 많은 질문을 응답해야 하고, 총 검사 시간은 12분 이내로 걸립니다. 
+ 만약 텍스트로 MBTI를 예측 할 수 있다면, 단순히 내가 SNS에서 올린 게시글, 카카오톡 데이터 만을 가지고 비교적 짧은 시간내에 성격을 검사를 할 수 있게 됩니다.

## 데이터 수집 
+ reference : https://shlee1990.tistory.com/864
+ R 크롤러를 활용, 디시인사이드의 MBTI별 갤러리별 게시물 3000개 수집 (https://gall.dcinside.com/mgallery/board/lists/?id=estj)
<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/data%ED%98%95%ED%83%9C.PNG"  width="100%">

## World Cloud 결과
image repos에 저장이 되어 있습니다.

몇개만 확인해 보겠습니다

### ESTJ 
<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/ESTJ.png"  width="40%">


### INTJ 
<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/INTJ.png"  width="40%">


### ESTP 
<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/ESTP.png"  width="40%">


## 분류 모형 만들기 
(pytorch로 처음 모델링 하려 했으나, colab의 cpuda gpu 문제가 계속 발생해 keras로 바꿨습니다.)

결과는 보시는 바와 같이 대실패 입니다

( 디시인사이드 갤러리에서 수집한 데이터 자체가 욕설, 정치적 글등 성격과 의미없는 데이터가 대부분이지 않았나 생각합니다.)

### Embedding layer를 활용한 MLP 분류 모델 
```{python}
embedding_dim =100

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.summary()
```
training acc는 높아지나 일반화 성능을 전혀 못함..

### training / validation acc 


<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/Embedding%20layer.png"  width="40%">

### LSTM + Embedding layer를 활용한 MLP 분류 모델 

### training/ validation acc 

```{python}
# build a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM

embedding_dim = 500

model_LSTM = Sequential()
model_LSTM.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model_LSTM.add(LSTM(64,  return_sequences=True))
model_LSTM.add(LSTM(64,return_sequences=True))
model_LSTM.add(LSTM(64,return_sequences=True))
model_LSTM.add(Flatten())
model_LSTM.add(Dense(32, activation='relu'))
model_LSTM.add(Dense(16, activation='softmax'))
model_LSTM.summary()

```

<img src="https://github.com/bongseokkim/natural-language-processing/blob/main/image/LSTM%2BEmbedding%20Layer.png"  width="40%">

