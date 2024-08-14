# 피해자_연관분석

## 기초환경설치
### package설치 (caret, dplyr, randomForest, ggplot2)
install.packages("dplyr")
install.packages("arules")
install.packages("arulesViz")
install.packages("RColorBrewer")

### library 불러오기
library(dplyr)
library(arules)
library(arulesViz)
library(RColorBrewer)
---------------------------------------------
---------------------------------------------
---------------------------------------------

## 기초 전처리
### 원본데이터 불러오기
# 코드를 실행하여 첨부한 원본데이터 (가해자차종 또는 피해자차종이 이륜차인 교통사고 정보(2015~2019년).csv)를 불러오기
data <- read.csv(file.choose())

### 필요한 변수 선택
# select()를 통해 원본데이터에서 사용할 변수들만 뽑아오기.
data.use <- select(data, 요일, 가해자법규위반, 노면상태, 기상상태, 가해당사자종별, 가해자신체상해정도, 가해자보호장구, 피해당사자종별, 피해자신체상해정도, 피해자보호장구)

### 신체상해정도(중상, 경상, 부상신고)를 부상으로 바꿔주기 => (사망, 부상, 상해없음)
# ifelse()를 통해 신체상해정도가 "중상, 경상, 부상신고"인 것들을 부상으로 바꾸고, 사망과 상해없음은 그대로 불러오기.
data.use$가해자신체상해정도 <- ifelse(data.use$가해자신체상해정도=="중상"|data.use$가해자신체상해정도=="경상"|data.use$가해자신체상해정도=="부상신고", "부상", data.use$가해자신체상해정도)
data.use$가해자신체상해정도 %>% table()
data.use$피해자신체상해정도 <- ifelse(data.use$피해자신체상해정도=="중상"|data.use$피해자신체상해정도=="경상"|data.use$피해자신체상해정도=="부상신고", "부상", data.use$피해자신체상해정도)
data.use$피해자신체상해정도 %>% table()

### 신체상해정도가 "기타불명"인 경우를 제외
# 신체상해정도 == "기타불명" =>  어느정도 다쳤는지 알 수 없기 때문에 filter()를 통해 분석에서 제거
data.use <- filter(data.use, data.use$가해자신체상해정도!="기타불명"&data.use$피해자신체상해정도!="기타불명")

### 변수들을 factor로 바꿔주기
#	결과변수와 설명변수들을 모두 as.factor()를 통해 변수타입을 factor로 바꾸기.
data.use$요일 <- data.use$요일 %>% as.factor()
data.use$가해자법규위반 <- data.use$가해자법규위반 %>% as.factor()
data.use$노면상태 <- data.use$노면상태 %>% as.factor()
data.use$기상상태 <- data.use$기상상태 %>% as.factor()
data.use$가해당사자종별 <- data.use$가해당사자종별 %>% as.factor()
data.use$가해자신체상해정도 <- data.use$가해자신체상해정도 %>% as.factor()
data.use$가해자보호장구 <- data.use$가해자보호장구 %>% as.factor()
data.use$피해당사자종별 <- data.use$피해당사자종별 %>% as.factor()
data.use$피해자신체상해정도 <- data.use$피해자신체상해정도 %>% as.factor()
data.use$피해자보호장구 <- data.use$피해자보호장구 %>% as.factor()
-----------------------------------------------
---------------------------------------------
---------------------------------------------
## 피해자 데이터 전처리
### 피해자당사자종별이 "이륜차"인 데이터 뽑기
# filter()를 통해 피해당사자종별이 이륜차인 데이터를 분리
data.use.ga <- filter(data.use, data.use$피해당사자종별 == "이륜차")

### 피해자신체상해정도(결과변수)가 "없음"인 경우 제거 
# 피해자신체상해정도 == "없음"인 데이터 제거 => 가해자가 혼자 사고 낸 경우에는 피해자가 없음, 그리하여 피해자의 상해정도가 "없음"으므로 분석에서 제거
data.use.del <- filter(data.use.del, data.use.del$피해자신체상해정도 != "없음")

### 결과변수의 타입을 character로 바꾼 후 factor로 변환
# 제거하고 난 후 factor의 다시 as.charactor후 as.factor하여 factor의 level을 줄이기 
data.use.pi$피해자신체상해정도 <- data.use.pi$피해자신체상해정도 %>% as.character()
data.use.pi$피해자신체상해정도 <- data.use.pi$피해자신체상해정도 %>% as.factor()

### 사용 할 변수만 뽑기
# "피해당사자종별" 제거 => 이미 피해당사자종별이 "이륜차"이므로 select()를 이용해 해당 변수 제거
data.apri <- select(data.use.del, 가해자신체상해정도, 피해자보호장구, 가해자법규위반, 요일, 가해당사자종별, 기상상태, 노면상태, 가해자보호장구, 피해자신체상해정도)

### 피해자보호장구가 "착용" or "미착용"인 데이터
# filter()를 통해 착용 or 미착용 데이터만 가져오기
data.apri <- filter(data.apri, data.apri$피해자보호장구=='착용'|data.apri$피해자보호장구=='미착용')

### 피해자보호장구 변수 타입을 character로 바꾼후 factor로 변환
# 착용 or 미착용만 가져왔기 때문에, factor의 레벨을 낮춰주기위해 as.character()후 다시 as.factor()
data.apri$피해자보호장구 <- data.apri$피해자보호장구 %>% as.character()
data.apri$피해자보호장구 <- data.apri$피해자보호장구 %>% as.factor()
--------------------------------
--------------------------------
--------------------------------

## 가해자 데이터 연관분석
### 연관규칙 만들기
# lhs => 피해자신체상해정도     rhs => 피해자보호장구
# 조건 => 최소부분집합 = 2, 최대부분집합 = 2, 최소지지도 = 0.001, 최소신뢰도 = 0.001
rules <- apriori(data.apri, control = list(verbose = T), parameter = list(minlen = 2, maxlen = 2,supp = 0.001, conf = 0.001),
                 appearance = list(rhs = c("피해자신체상해정도=사망","피해자신체상해정도=부상","피해자신체상해정도=상해없음"),
                                   lhs = c("피해자보호장구=착용","피해자보호장구=미착용")))

### 향상도(lift) 기준으로 정렬
rules.sort <- sort(rules, by = "lift")

### 정렬된 규칙 확인
inspect(rules.sort)

### 연관규칙을 시각화로 보기
# 매트릭스로 확인
plot(rules.sort, method = "matrix", control = list(type = "items",col=brewer.pal(11,"Spectral")), engine = "htmlwidget")
