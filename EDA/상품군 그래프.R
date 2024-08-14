library(dplyr)
library(ggplot2)
library(reshape2)
library(scales)
### 데이터 불러오기 ###
rm(list=ls())
data.use <- read.csv(choose.files())
data.use %>% head()

##### 분기 별 온라인 쇼핑 합계액 #####
data.use.1 <- data.use %>% dplyr::select(시점,합계)
names(data.use.1) <- c("분기","(억)원")
data.use.1 %>% head()
ggp <- ggplot(data=data.use.1,aes(x=분기,y=`(억)원`,group=1)) +
  geom_line(size=2,color="skyblue")+ 
  scale_y_continuous(labels=comma) +
  theme(axis.text=element_text(size=30),axis.title=element_text(size=25,face="bold"),axis.text.x=element_text(angle=45,hjust=1)) 
ggp

##### 분기 별 온라인쇼핑 상품군 거래액 #####
data.use.2 <- data.use %>% dplyr::select(시점,가전.전자.통신기기,생활용품,음식서비스,여행.및.교통서비스,음.식료품)
data.use.2 %>% head()
names(data.use.2) <- c("시점","가전전자통신기기","생활용품","배달음식서비스","여행및교통서비스","음식료품")

data.melt.2 <- melt(data.use.2,id.var="시점",measure.vars=c("가전전자통신기기","생활용품","배달음식서비스","여행및교통서비스","음식료품"))
names(data.melt.2) <- c("분기","상품군","(억)원")
data.melt.2 %>% summary()
data.melt.2 %>% head()

## 데이터 전처리 ##
data.melt.a <- data.melt.2 %>% filter(상품군=='가전전자통신기기')
data.melt.a$prepro <- data.melt.a$`(억)원`/data.melt.a[1,3]*100

data.melt.b <- data.melt.2 %>% filter(상품군=='생활용품')
data.melt.b$prepro <- data.melt.b$`(억)원`/data.melt.b[1,3]*100

data.melt.c <- data.melt.2 %>% filter(상품군=='배달음식서비스')
data.melt.c$prepro <- data.melt.c$`(억)원`/data.melt.c[1,3]*100

data.melt.d <- data.melt.2 %>% filter(상품군=='여행및교통서비스')
data.melt.d$prepro <- data.melt.d$`(억)원`/data.melt.d[1,3]*100

data.melt.e <- data.melt.2 %>% filter(상품군=='음식료품')
data.melt.e$prepro <- data.melt.e$`(억)원`/data.melt.e[1,3]*100

data.rb <- rbind(data.melt.a,data.melt.b,data.melt.c,data.melt.d,data.melt.e)
data.rb %>% tail()

## 데이터 시각화 ##
ggp <- ggplot(data=data.rb,aes(x=분기,y=`prepro`,group=상품군,color=상품군)) +
  geom_line(size=2) + labs(x="분기", y="변화율(%)") + 
  scale_y_continuous(breaks=c(100,300,500,700)) +
  theme(axis.text=element_text(size=25),axis.title=element_text(size=25,face="bold"),axis.text.x=element_text(angle=45,hjust=1)) +
  theme(legend.title=element_text(size=25,face="bold"),legend.text=element_text(size=20))
ggp


