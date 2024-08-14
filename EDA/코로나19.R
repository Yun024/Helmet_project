# 코로나19

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

## 데이터 불러오기
# 파일명 : "WHO-COVID-19-global-data.csv" 불러오기
data <- read.csv(choose.files(),)   
data %>% head()

## 데이터 전처리
### 컬럼명 변경
# data의 각 컬럼명을 괄호안의 이름으로 변경
names(data) <- c("날짜","국가코드","국가","WHO지대","확진자","누적확진자수","사망자","누적사망자수")

### 한국 데이터 추출
# data의 `국가코드`열에서 KR로 되어있는 행을 추출하여 data.use에 저장
data.use <- data %>% filter(국가코드=="KR")

### `날짜`의 데이터 타입 변경
# data.use의 character로 되어있는 `날짜`열을 Date로 변경
data.use$날짜 <- data.use$날짜 %>% as.Date()

### 데이터 시각화
# data.use를 이용해 x축은 `날짜` y축은 `누적확진자수`로 되어있는 1개의 꺽은선 그래프를 그리기.
# 꺽은선의 크기는 2, 색은 다크블루로 지정, 그리고 축제목을 x축은 `날짜` y축은 `확진자수`로 지정
# x축의 레이블 범위를 2020-01-19에서 2020-09-06에서 매월 19일로 지정
# x축 레이블 크기를 25, 각도를 30도로 지정, 축제목 글씨포인트를 25로 하고 굵기를 진하게 지정 
# y축 레이블 크기를 25, 축제목 글씨포인트를 25로 하고 굵기를 진하게 지정
ggp <- ggplot(data=data.use,aes(x=날짜,y=누적확진자수,group=1)) +
  geom_line(size=2,color="darkblue") + labs(x="날짜", y="확진자수") +
  scale_x_date(breaks=seq(as.Date("2020-01-19"),as.Date("2020-09-06"),by="1 month")) +
  scale_y_continuous(labels=comma) +							 : y축의 레이블에 콤마를 찍어준다. 
  theme(axis.text.x=element_text(size=25,angle=30,hjust=1),axis.title=element_text(size=25,face="bold")) +
  theme(axis.text.y=element_text(size=25),axis.title=element_text(size=25,face="bold"))
ggp 
