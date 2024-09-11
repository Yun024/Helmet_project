# 대전지역 내 동별로 EPDO지수를 구해 가장 높은 상위 5곳을 뽑아 영상처리를 위한 동영상을 직접 촬영한다.
# EPDO = (12*사망사고건수) + (3*부상사고건수) + 물피 사고  

#패키지 설치(dplyr, writexl, readxl, ggmap)
install.packages("dplyr")
install.packages("writexl")
install.packages("readxl")
install.packages("ggmap")

#library 불러오기
library(dplyr)
library(writexl)
library(readxl)
library(ggmap)

# 사망사고건수 구하기

#file.choose()는 파일이 저장된 위치에서 바로 데이터를 선택해 불러올 수 있다.
# file.choose()를 이용해 data라는 이름을 지정해서 read.csv()로 csv파일을 불러온다.
data <- read.csv(file.choose()) 
# data에 자료가 정상적으로 입력됐는지 확인하기 위해 head()를 이용한다.
data %>%  head()
# data1에 원하는 변수 만을 추출하기 위해 filter()를 이용해 대전에서 발생한 사망사고 발생 건을 불러온다.
data1<-filter(data,가해자신체상해정도=="사망"&발생지_시도=="대전")
data1 %>% head()

# 동별 사망사고 건수를 파악하기 위해 data2에 data1의 법정동명별로 group_by()로 묶고,
# 그룹 별로 행의 개수를 세기 위해서 summarise(n=n())를 이용한다. 
data2<-group_by(data1,법정동명) %>% summarise(count=n())
View(data2)

# 부상사고건수 구하기

# file.choose()를 이용해 data3라는 이름을 지정해서 read.csv()로 csv파일을 불러온다.
data3 <- read.csv(file.choose())
# data3에 자료가 정상적으로 입력됐는지 확인하기 위해 head()를 이용한다.
data3 %>%  head()
# data4에 원하는 변수 만을 추출하기 위해 filter()를 이용해 부상(중상,경상,부상신고)사고 발생 건을 불러온다.
data4<-filter(data3,가해자신체상해정도=="중상"|가해자신체상해정도=="경상"|가해자신체상해정도=="부상신고")
# data5에 원하는 변수 만을 추출하기 위해 filter()를 이용해 대전지역에서 발생한 건을 불러온다.
data5<-filter(data4,발생지_시도=="대전")
data5 %>% head()

# 동별 부상사고건수를 파악하기 위해 data6에 data5의 법정동명별로 group_by()로 묶고,
# 그룹 별로 행의 개수를 세기 위해서 summarise(n=n())를 이용한다. 
data6<-group_by(data5,법정동명) %>% summarise(n=n())
View(data6)

# 컬럼명 변경
names(data6) <- c("법정동명","count")

# Dong에 행을 기준으로 rbind()를 이용해 data2(사망사고건수), data6(부상사고건수)을 결합한다.
Dong<-rbind(data2,data6)

# 엑셀에서 함수계산을 통해 EXPO지수를 계산하기 위해서 Dong 데이터를 내 위치에 xlsx파일로 저장한다.

# write_xlsx()를 이용해 r데이터를 xlsx파일로 저장한다.
# Dong이라는 r데이터를 xlsx파일로 저장하고, path를 원하는 위치로 지정해서 파일명을 "동별EPDO.xlsx"로 저장한다. 
write_xlsx(Dong,path="동별score.xlsx")



# 엑셀에서 함수를 통해 EPDO지수를 구한다.
# (사망사고건이 없는 동네는 부상이 없고, 부상사고건이 없는 동네는 사망이 있는 경우가 반 이상이기 때문에
# 사망사고가 있는 동네는 부상사고건수를 0으로 , 부상사고가 있는 동네는 사망사고건수를 0으로 지정해 계산한다.)
# 동별로 지도에 점을 찍기 위해 구 이름과 위치정보(위도,경도)를 구글링을 통해서 입력한다.

# 이렇게 생성한 엑셀 데이터를 "동별사고score.xlsx"라고 저장하고 불러온다.



##################################
# 지도 전처리 파일 불러오기
# Dong에 file.choose()를 통해 파일을 저장한 위치에서 엑셀 데이터를 불러온다.
# 기존의 동별사고score.xlsx가 아닌 구와 위도, 경도를 넣은 "동별사고score_위도경도.xlsx"로 대체
Dong<-read_excel(file.choose())



# 지도 작성 전 전체적인 위치를 지정해주기 위해 boxLocation에 경도,위도를 지정한다.
boxLocation<-c(127.378953,36.321655)

# register_google()로 구글 맵에서 받은 인증 key를 등록한다.
from dotenv import load_datenv
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

register_google(key=openai_api_key)

# get_map()을 통해 KrMap에 지도 위치를 boxLocation로 지정하고 타입은 "roadmap", 출처는 "google"이고 지도를 +11만큼 확대하여 보여준다.
KrMap<-get_map(boxLocation, maptype="roadmap",source="google",zoom=11)
# ggmap()을 통해 KrMap을 실행한다.
ggmap(KrMap)

# ggmap()을 통해 실행시킨 지도위에 점을 찍는다.
# 점 색상으로 구를, 점 크기로 EPDO지수를 구분한다. 
# 지도 구성 요소들의 디자인을 변경하는 theme 함수를 이용해 한 눈에 볼 수 있도록 범례 위치를 지도 안 쪽으로 위치시킨다.
ggmap(KrMap)+
  geom_point(data=Dong,aes(경도,위도,color=시군구,size=EPDO))+
  theme(legend.position = c(0.1,0.6))
