# 동영상 분석 
- 안전모 착용률을 높이기 위한 방법으로 헬멧 착용여부를 파악할 수 있는 Detecting기술 선정
- Open image dataset에서 train데이터와 test데이터 사용
- Image Detected , Video Detected, Real Time Detected
  
## YOLO 알고리즘
- R-CNN은 정확성이 높지만, 이미지를 여러 개로 쪼개어 훈련하기 때문에 많은 시간이 소요된다
- YOLO는 이미지 한개를 그대로 훈련하기 때문에 R-CNN에 비해 적은 시간을 소요한다
- 우린느 실시간 운전자 안전모 착용여부 등 실시간 객체 인식이 필요하기 때문에 YOLO알고리즘 채택
<img src="https://github.com/user-attachments/assets/f3069d86-d042-4eb2-8157-284c4035fe59"  width="600" height="350"/><br>

### 데이터  *[바로가기](https://github.com/Yun024/helmet_project/blob/main/Video%20Analysis/crawling.py)*
- Data 다운로드 `Helmet 400`, `Human Hair 400`, `Label csv`
- Process `train data path`, `test data path`, `txt로 추출하여 9:1로 나누기`
- anchor 추출 `kmeans로 여러 데이터들의 label된 박스 위치를 kmeans로 뽑아냄`
- GPU사용을 위해 CUDA설치
  
### YOLO모델 Darknet버전 *[바로가기](https://github.com/Yun024/helmet_project/blob/main/Video%20Analysis/Helmet_detection_YOLOV3.py)*
- Download
- training
  * cfg: `filter = (2+5) *3`, `class = 2`, `anchor`
  * data: `train/test path`, `weight path`, `class=2`
  * names: `0 = Human hair = Danger`, `1 = Helmet = Safe`
  * weight: `1000번 반복마다 생성`
<img src="https://github.com/user-attachments/assets/5b5a97a4-7dab-4c81-86c7-c07fcd878815"  width="700" height="300"/><br>


    
- 인식결과 
  * `이미지`, `동영상`, `실시간` 인식 : threshold로 얼마나 확실한 box를 표시할건지 (0~1사이)
<img src="https://github.com/user-attachments/assets/16ae09bf-220c-4486-954a-9bc067bc8401"  width="700" height="300"/><br>



### Tracking *[바로가기](https://github.com/Yun024/helmet_project/blob/main/Video%20Analysis/yolov3_deepsort_fuckingurass.py)*
- 사용이유 : 차량을 세기위해
- weight를 텐서플로우 파일로 변환
- 동영상 트래킹
   * 각 객체 ID를 dictionary자료형에 저장
   * set으로 중복 제거 후 길이 출력(count)


  

