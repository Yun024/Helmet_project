# 가구추계

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
### 변수 초기화
rm(list=ls()

### 데이터 불러오기
# 파일명 : "가구주의_연령_가구유형_가구원수별_추계가구전국_20200831151205.csv" 불러오기
data <- read.csv(choose.files(),)   
data %>% head()

### 데이터 전처리
data.use <- data %>% select(시점,X1인,X4인)	:  data의 `시점`,`X1인`,`X4인`열을 선택하여 data.use에 저장

### 변수 가공 및 소수점 제거
# "X1인",  "X4인"을 1000으로 나눈 후, trunc()를 이용해 소숫점 제거
data.use$X1인 <- trunc(data.use$X1인/1000)
data.use$X4인 <- trunc(data.use$X4인/1000)

### columns명 변경
# 변수명 정리(연도, 1인, 4인)
names(data.use) <- c("연도","1인","4인")

### 2034년 이전 데이터 추출
# filter()를 이용해 "연도" 변수에서 2034년보다 작은 자료만 추출
data.use <- data.use %>% dplyr::filter(연도<2034)

### 데이터 재구조화
# melt()를 이용해 재구조화하여 "data.melt"에 저장
# 식별자변수 = "연도"       측정변수 = "1인","4인" 
data.melt <- melt(data.use,id.var="연도",measure.vars=c("1인","4인")) 

### columns명 변경
# 변수명 정리(연도, 가구원수, 가구(천))
names(data.melt) <- c("연도","가구원수","가구(천)")

### 요약값 확인
data.melt %>% summary()

### 데이터 시각화
# data.melt의 `연도`를 x축, `가구(천)`을 y축, 색과 그룹을 `가구원수`로 하는 ggplot을 생성
# 꺽은선 그래프의 사이즈는 2로 지정
# x축의 범위를 2020,2025,2030으로 지정 
# y축의 범위를 2500,4000,5500,7000으로 지정하고 콤마를 찍어주는 역할 
# 레이블의 크기를 30으로 지정, 축제목의 크기를 25로 지정하고 굵기를 진하게 지정
# 범례의 글씨 크기를 25로 지정하고 글씨를 굵게 지정, 범례제목 크기를 20으로 지정
ggp <- ggplot(data=data.melt,aes(x=연도,y=`가구(천)`,group=가구원수,color=가구원수)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=c(2020,2025,2030)) +
  scale_y_continuous(breaks=c(2500,4000,5500,7000),labels=comma) + 
  theme(axis.text=element_text(size=30),axis.title=element_text(size=25,face="bold")) +
  theme(legend.title=element_text(size=25,face="bold"),legend.text=element_text(size=20))
ggp 


