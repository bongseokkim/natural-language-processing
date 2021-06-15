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

#### 프로젝트 소개
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

## World Cloud 결과
