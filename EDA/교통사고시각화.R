# 교통사고시각화

## 기초환경설치
### package설치 (dplyr, reshape2, ggplot2, scales)
install.packages("dplyr")
install.packages("reshape2")
install.packages("ggplot2")
install.packages("scales")

### library 불러오기
library(dplyr)
library(reshape2)
library(ggplot2)
library(scales)
---------------------------------------------
---------------------------------------------
---------------------------------------------

#### 변수 초기화
rm(list=ls()

### 데이터 불러오기
# 파일명 : "가해운전자 차종별 월별 교통사고(2010~2019).csv" 불러오기
data <- read.csv(choose.files(),)   
data %>% head()

##  차종별 사고 변화율
### 차종별 사망자의 변화율을 년도에 따른 꺽은선그래프
### 데이터 추출
# data.use의 `가해운전자.차종별`변수가 `사고건수`인 행만 추출한 후 `가해운전자.차종별`를 제거한 데이터를 data.use.1에 저장
data.use.1 <- data.use %>% filter(가해운전자.차종별=="사고건수") %>% select(-가해운전자.차종별)

### 데이터 재구조화
# 그래프를 그리기 위해 data.use.1을 melt함수를 이용해 재구조화 시키고 data.melt에 저장
data.melt <- melt(data.use.1,id.var="사고년도",measure.vars=c("승용차","승합차","이륜차","화물차"))

### 변수명 변경
# 변수명을 `년도`,`차량분류`,`사고건수`로 변경
names(data.melt) <- c("년도","차량분류","사고건수")

### 차량분류별 데이터 셋 추출 & "각 사고건수/2010년사고건수" 인 변수 만들기
# 차량분류 변수가 `승용차`인 행만 추출해서 data.melt.a에 저장 및 "prepro" 변수 생성
# 차량분류 변수가 `승합차`인 행만 추출해서 data.melt.b에 저장 및 "prepro" 변수 생성
# 차량분류 변수가 `이륜차`인 행만 추출해서 data.melt.c에 저장 및 "prepro" 변수 생성
# 차량분류 변수가 `화물차`인 행만 추출해서 data.melt.d에 저장 및 "prepro" 변수 생성
data.melt.a <- data.melt %>% filter(차량분류=='승용차')
data.melt.a$prepro <- data.melt.a$사고건수/data.melt.a[1,3]*100
data.melt.b <- data.melt %>% filter(차량분류=='승합차')
data.melt.b$prepro <- data.melt.b$사고건수/data.melt.b[1,3]*100
data.melt.c <- data.melt %>% filter(차량분류=='이륜차')
data.melt.c$prepro <- data.melt.c$사고건수/data.melt.c[1,3]*100
data.melt.d <- data.melt %>% filter(차량분류=='화물차')
data.melt.d$prepro <- data.melt.d$사고건수/data.melt.d[1,3]*100

### 데이터 병합
# 나눠놓은 데이터를 rbind를 통해 데이터를 통합시키고 data.rb에 저장
data.rb <- rbind(data.melt.a,data.melt.b,data.melt.c,data.melt.d)

### 데이터 시각화
# data.rb의 `년도`를 x축, `prepro`을 y축, 색과 그룹을 `차량분류`로 하는 ggplot을 생성 
# 꺽은선의 사이즈를 2로, x축제목을 `년도`, y축제목을 `변화율(%)`로 지정
# y축의 범위를 80,100,12,140,160으로 지정
# x축의 범위를 2010,2013,2016,2019로 지정
# x축과 y축의 레이블의 크기를 30으로, 축제목으로 25, 굵기는 진하게 지정
# 범례의 제목을 25사이즈, 굵기는 진하게 범례 속 글씨는 20사이즈로 지정
ggp <- ggplot(data=data.rb,aes(x=년도,y=prepro,group=차량분류,color=차량분류)) +
  geom_line(size=2) +   labs(x="년도", y="변화율(%)")+
  scale_y_continuous(breaks=c(80,100,120,140,160)) +
  scale_x_continuous(breaks=c(2010,2013,2016,2019))+
  theme(axis.text=element_text(size=30),axis.title=element_text(size=25,face="bold")) +
  theme(legend.title=element_text(size=25,face="bold"),legend.text=element_text(size=20))
ggp
---------------------------
---------------------------
---------------------------

## 차종별 사망률 막대그래프
### 사망자수 list 만들기 => data.death
# data.death라는 list형태로 되어 있는 빈 데이터를 생성
# data.use에서 변수 "가해운전자.차종별"==`사망자수`로 되어있는 `승용차`변수의 모든 값들을 합하고 data.death의 1열에 저장
# data.use에서 변수 "가해운전자.차종별"==`사망자수`로 되어있는 `승합차`변수의 모든 값들을 합하고 data.death의 2열에 저장
# data.use에서 변수 "가해운전자.차종별"==`사망자수`로 되어있는 `이륜차`변수의 모든 값들을 합하고 data.death의 3열에 저장
# data.use에서 변수 "가해운전자.차종별"==`사망자수`로 되어있는 `화물차`변수의 모든 값들을 합하고 data.death의 4열에 저장
data.death <- list()
data.death[1] <- data.use %>% filter(가해운전자.차종별=="사망자수") %>% select(승용차) %>% sum()
data.death[2] <- data.use %>% filter(가해운전자.차종별=="사망자수") %>% select(승합차) %>% sum()	
data.death[3] <- data.use %>% filter(가해운전자.차종별=="사망자수") %>% select(이륜차) %>% sum()
data.death[4] <- data.use %>% filter(가해운전자.차종별=="사망자수") %>% select(화물차) %>% sum()

### 부상자수 list 만들기
# data.inj라는 list형태로 되어 있는 빈 데이터를 생성
# data.use에서 변수 "가해운전자.차종별"==`부상자수`로 되어있는 `승용차`변수의 모든 값들을 합하고 data.inj의 1열에 저장
# data.use에서 변수 "가해운전자.차종별"==`부상자수`로 되어있는 `승합차`변수의 모든 값들을 합하고 data.inj의 2열에 저장
# data.use에서 변수 "가해운전자.차종별"==`부상자수`로 되어있는 `이륜차`변수의 모든 값들을 합하고 data.inj의 3열에 저장
# data.use에서 변수 "가해운전자.차종별"==`부상자수`로 되어있는 `화물차`변수의 모든 값들을 합하고 data.inj의 4열에 저장
data.inj <- list()
data.inj[1] <- data.use %>% filter(가해운전자.차종별=="부상자수") %>% select(승용차) %>% sum()
data.inj[2] <- data.use %>% filter(가해운전자.차종별=="부상자수") %>% select(승합차) %>% sum()
data.inj[3] <- data.use %>% filter(가해운전자.차종별=="부상자수") %>% select(이륜차) %>% sum()
data.inj[4] <- data.use %>% filter(가해운전자.차종별=="부상자수") %>% select(화물차) %>% sum() 

### 이름변경을 위한 변수
# 전처리완성데이터에 넣을 차량변수에 해당하는 이름을 생성하여 "data.name"로 지정
data.name <- c("승용차","승합차","이륜차","화물차")

### 최종데이터
# data.acc라는 이름을 가진 빈 데이터 프레임 생성
# 비어있는 data.acc에 data.name, data.death, data.inj를  열 기준으로 병합 # cbind()
# data.acc를 데이터 프레임형태로 변형 # as.data.frame()
# 변수명을 `차량분류`,`사망자수`,`부상자수`로 지정
# `차량분류` 변수를 문자열로 변경해준 뒤 factor형태로 변경
# `사망자수`변수를 정수 형태로 변경
# `부상자수`변수를 정수 형태로 변경
# `사망자수`와 `부상자수`를 합친 값을 `신고자수`라는 변수에 저장
# `사망자수`를 `신고자수`변수로 나눈 값을 `사망률`변수에 저장하여 최종데이터 완성
data.acc <- data.frame()
data.acc <- cbind(data.name,data.death,data.inj)
data.acc <- data.acc %>% as.data.frame
names(data.acc) <- c("차량분류","사망자수","부상자수")
data.acc$차량분류 <- data.acc$차량분류 %>% as.character() %>% as.factor()
data.acc$사망자수 <- data.acc$사망자수 %>% as.integer()
data.acc$부상자수 <- data.acc$부상자수 %>% as.integer()
data.acc$신고자수 <- data.acc$사망자수 + data.acc$부상자수
data.acc$사망률 <- data.acc$사망자수/data.acc$신고자수

### 데이터 시각화
# data.acc를 이용한 ggplot생성 
# x축은 `차량분류` y축은 `사망률`로하고 그룹은 `차량분류`의 4가지로 나뉘는 막대그래프를 생성한다.
# x축제목은 `차종` y축제목은 `사망률`로 지정 
# 그래프의 레이블의 크기는 30으로 지정, 그래프의 축제목 크기는 25로, 굵기는 진하게 지정한다.  
# 범례를 제거해준다
ggplot(data.acc)+
  geom_bar(aes(x = 차량분류, y = 사망률, fill = 차량분류), stat = 'identity')+
  labs(x="차종", y="사망률")+
  theme(axis.text=element_text(size=30),axis.title=element_text(size=25,face="bold")) +
  theme(legend.position="none")
