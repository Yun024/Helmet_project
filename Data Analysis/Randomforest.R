## #기초환경설치
### package설치 (caret, dplyr, randomForest, ggplot2)
install.packages("caret")
install.packages("dplyr")
install.packages("randomForest")
install.packages("ggplot2")

### library 불러오기
library(caret)
library(dplyr)
library(randomForest)
library(ggplot2)
---------------------------------------------
---------------------------------------------
---------------------------------------------

### 기초 전처리

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

### 가해자 데이터 전처리

### 가해자당사자종별이 "이륜차"인 데이터 뽑기
# filter()를 통해 가해당사자종별이 이륜차인 데이터를 분리
data.use.ga <- filter(data.use, data.use$가해당사자종별 == "이륜차")

### 결과변수 각 카테고리별 데이터 셋 분리
# 각 항목별 건수를 비슷하게 맞춰주기 위해서 먼저 사망, 부상, 상해없음 별로 데이터를 인덱싱을 통해 분리
dam_1 <- data.use.ga[data.use.ga$가해자신체상해정도=="사망",]
dam_2 <- data.use.ga[data.use.ga$가해자신체상해정도=="상해없음",]
dam_3 <- data.use.ga[data.use.ga$가해자신체상해정도=="부상",]

### oversample ("부상"카테고리의 수만큼 oversample) 및 일정한 표본을 뽑기위해 set.seed로 설정
# 코드를 재실행시 결과들이 동일하게 나올수 있도록 set.seed를 사용하여 일정한 값을 주기
set.seed(822)
# 범주들 중 가장 많은 갯수가 속해있는 "부상"카테고리 수인 45066개로 dplyr패키지의 sample_n()를 통해 oversampling.
# replace = TRUE => (사망과 상해없음에 속해있는 데이터셋의 수가 45066개 보다 적기 때문에 복원추출을 하는 옵션)
dam_sam1 <- sample_n(dam_1 , 45066, replace = T) # 사망인 데이터 oversampling
dam_sam2 <- sample_n(dam_2 , 45066, replace = T) # 상해없음인 데이터 oversampling
dam_sam3 <- sample_n(dam_3 , 45066, replace = T) # 부상인 데이터 oversampling

### oversampling한 데이터 병합시키기
# 위에 코드에서 oversampling한 데이터들을 rbind()를 통해 병합
data.use.ga.bind <- rbind(dam_sam1, dam_sam2, dam_sam3)
# randomforest분석을 할 때마다 임의로 train/ test셋이 분리되기 때문에 별도의 train / test셋은 필요하지 않습니다.
---------------------------------------------
---------------------------------------------
---------------------------------------------

### 가해자 randomforest 분석
### randomforest를 이용한 가해자모델만들기
# randomforest를 실행할 경우 임의로 train/test으로 나뉘며 train으로 모델을 만들기 때문에 set.seed함수를 통해
재실행 후에도 동일한 결과를 나타나게 함
# 목표변수 = "가해자신체상해정도" 
# 설명변수 = 요일, 가해자법규위반, 노면상태, 기상상태, 가해자보호장구, 피해당사자종별, 피해자신체상해정도, 피해자보호장구 
# 세부옵션 = ntree( 몇개의 나무를 생성할지 정하는 옵션 ) => 의사결정나무의 갯수를 300개 
# importance ( 변수의 상대적 중요도를 보기위한 옵션)  => importance = T로 설정
fitRF_ga <- 
  randomForest(가해자신체상해정도~요일+가해자법규위반+노면상태+기상상태+가해자보호장구+
                          피해당사자종별+피해자신체상해정도+피해자보호장구, data = data.use.ga.bind, importance = TRUE, ntree = 300)
fitRF_ga
# 실행시 나오는 obb error는 모델을 만드는 train이 아닌 모델에 쓰이지 않는 test을 통해 평가한 오차 => oob error = 26.14%
# accuracy(정분류율)은 "1-oob error" => accuracy = 0.7386 

### 가해자 그래프 그리기 설정
# varImpPlot()를 통해 변수중요도 그래프를 지정 by. randomForest 패키지
# as.data.frame()로 데이터프레임으로 변경
# rownames을 varnames라는 변수 생성, rownames을 제거
# 강조하고 싶은 "가해자보호장구" 에 해당하는 위치에만 2, 나머지는 1이라는 값 추가
imp <- varImpPlot(fitRF_ga)
imp <- as.data.frame(imp)
imp$varnames <- rownames(imp) # row names to column
rownames(imp) <- NULL  
imp$var_categ <- c(1,1,1,1,2,1,1,1)

### 가해자 그래프 실행하기
# ggplot()을 이용해 변수중요도 그래프 생성 by. ggplot2 패키지
ggplot(imp, aes(x=reorder(varnames, MeanDecreaseAccuracy), weight=MeanDecreaseAccuracy)) +
  geom_bar()+
  geom_bar(data = subset(imp, var_categ=="2"), fill = "#ff0000")+
  scale_fill_discrete(name="Variable Group") +
  ylab("MeanDecreaseAccuracy")+
  xlab("")+
  coord_flip()+
  theme(axis.text.y = element_text(size = 20, face = "bold"))
---------------------------------------
---------------------------------------------
---------------------------------------------

### 피해자 데이터 전처리

### 피해자당사자종별이 "이륜차"인 데이터 뽑기
# filter()를 통해 피해당사자종별이 이륜차인 데이터를 분리
data.use.pi <- filter(data.use, data.use$피해당사자종별 == "이륜차")

### 피해자신체상해정도(결과변수)가 "없음"인 경우 제거 
# 피해자신체상해정도 == "없음"인 데이터 제거 => 가해자가 혼자 사고 낸 경우에는 피해자가 없음, 그리하여 피해자의 상해정도가 "없음"으므로 분석에서 제거
data.use.pi <- filter(data.use.pi, data.use.pi$피해자신체상해정도 != "없음")

### 결과변수의 타입을 character로 바꾼 후 factor로 변환
# 제거하고 난 후 factor의 다시 as.charactor후 as.factor하여 factor의 level을 줄이기 
data.use.pi$피해자신체상해정도 <- data.use.pi$피해자신체상해정도 %>% as.character()
data.use.pi$피해자신체상해정도 <- data.use.pi$피해자신체상해정도 %>% as.factor()

### 결과변수 각 카테고리별 데이터 셋 분리
# 각 항목별 건수를 비슷하게 맞춰주기 위해서 먼저 사망, 부상, 상해없음 별로 데이터를 인덱싱을 통해 분리
dam_1 <- data.use.pi[data.use.pi$피해자신체상해정도=="사망",]
dam_2 <- data.use.pi[data.use.pi$피해자신체상해정도=="상해없음",]
dam_3 <- data.use.pi[data.use.pi$피해자신체상해정도=="부상",]

### oversample ("부상"카테고리의 수만큼 oversample) 및 일정한 표본을 뽑기위해 set.seed로 설정
# 코드를 재실행시 결과들이 동일하게 나올수 있도록 set.seed() 설정 
# 범주들 중 가장 많은 갯수가 속해있는 "부상"카테고리 수인 45066개로 dplyr패키지의 sample_n 함수를 통해 oversampling 
# replace = TRUE는 사망과 상해없음에 속해있는 데이터셋의 수가 45066개 보다 적기 때문에 복원추출을 하기 위해 추가
# 일정한 샘플을 뽑기위해 set.seed() 설정
set.seed(822)
# oversampling한 데이터들을 rbind 함수를 통해 병합.
dam_sam1 <- sample_n(dam_1 , 45066, replace = T) # 사망인 데이터 oversampling
dam_sam2 <- sample_n(dam_2 , 45066, replace = T) # 상해없음인 데이터 oversampling
dam_sam3 <- sample_n(dam_3 , 45066, replace = T) # 부상인 데이터 oversampling

### oversampling한 데이터 병합시키기
data.use.pi.bind <- rbind(dam_sam1, dam_sam2, dam_sam3)
# randomforest분석을 할 때마다 임의로 train/ test셋이 분리되기 때문에 별도의 train / test셋은 필요하지 않습니다.
------------------------------
---------------------------------------------
---------------------------------------------

### 피해자 randomforest 분석

### randomforest를 이용한 피해자모델만들기
# randomforest를 실행할 경우 임의로 train/test으로 나뉘며 train으로 모델을 만들기 때문에 set.seed함수를 통해 재실행 후에도 동일한 결과를 나타나게 함
# 목표변수 = "피해자신체상해정도" 
# 설명변수 = 요일, 가해자법규위반, 노면상태, 기상상태, 가해자보호장구, 피해당사자종별, 가해자신체상해정도, 피해자보호장구 
# 세부옵션 = ntree( 몇개의 나무를 생성할지 정하는 옵션 ) => 의사결정나무의 갯수를 300개 
# importance ( 변수의 상대적 중요도를 보기위한 옵션)  => importance = T로 설정
fitRF_pi <- 
  randomForest(피해자신체상해정도~요일+가해자법규위반+노면상태+기상상태+가해당사자종별+가해자보호장구
                          +가해자신체상해정도+피해자보호장구, data = data.use.pi.bind, importance = TRUE, ntree = 300)

fitRF_pi

### 피해자 그래프 그리기 설정
# varImpPlot()를 통해 변수중요도 그래프를 지정 by. randomForest 패키지
# as.data.frame()로 데이터프레임으로 변경
# rownames을 varnames라는 변수 생성, rownames을 제거
# 강조하고 싶은 "피해자보호장구" 에 해당하는 위치에만 2, 나머지는 1이라는 값 추가
imp <- varImpPlot(fitRF_pi)
imp <- as.data.frame(imp)
imp$varnames <- rownames(imp) # row names to column
rownames(imp) <- NULL  
imp$var_categ <- c(1,1,1,1,1,1,1,2)

### 피해자 그래프 실행하기
ggplot(imp, aes(x=reorder(varnames, MeanDecreaseAccuracy), weight=MeanDecreaseAccuracy)) +
  geom_bar()+
  geom_bar(data = subset(imp, var_categ=="2"), fill = "#ff0000")+
  scale_fill_discrete(name="Variable Group") +
  ylab("MeanDecreaseAccuracy")+
  xlab("")+
  coord_flip()+
  theme(axis.text.y = element_text(size = 20, face = "bold"))
--------------------
--------------------------------------
--------------------------------------
