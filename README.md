# object-detection

# 목록
1. Faster R-CNN
2. SSD
3. yolov1 아키텍쳐 직접 구현
4. yolov1 아키텍쳐 data augmentation

---
# 1. Faster R-CNN
# Object Detection(객체 탐지)
객체 탐지는 컴퓨터 비전과 이미지 처리와 관련된 기술로서, 디지털 이미지와 비디오로 특정한 계열의 시맨틱 객체 인스턴스를 감지하는 일을 다룬다.

# Faster R-CNN(NIPS 2015)
Faster R-CNN은 하나의 unified network로 Detection과제를 수행한다.

feature maps는 RPN을 통과해서 바운딩 박스를 출력한다. 그 후에 바운딩 박스에 해당하는 feature maps는 크롭해서 ROI Pooling과 classificaion을 수행한다. 

![](https://velog.velcdn.com/images/seonydg/post/0b0b7005-3640-4f36-b230-b40eff5fc786/image.png)

RPN은 이미지를 아래와 같이 빨간색 격자 무늬로 쪼갠 후, 파란색처럼 앵커 박스라는 개념을 도입한다. 빨간색 박스 하나당 미리 정의된 앵커 박스 여러 개가 있고, 앵커 박스 안에 object가 있다고 가정한(가장 알맞다고 예상되는) 박스를 찾는다.
앵커 박스의 크기가 고정되어 있기에 논문에서는 3가지 방법을 제시하는데, 첫번째는 이미지 자체의 크기(scale)를 줄여가며 다음은 필터의 크기를 3*3이나 7*7 등 물체의 크기에 맞춰 검출되도록 조절하는 것이고 마지막으로 레퍼런스(앵커 박스처럼 해당 물체에 적합한 박스들을 미리 여러개의 사이즈로 구분)를 여러 개를 사용한다.

![](https://velog.velcdn.com/images/seonydg/post/43bd02ce-4fb1-43af-ab8e-1f6419d4a6bd/image.png)

RPN은 sliding window를 정의하고 feature maps을 움직이며 계산을 하고 하나의 sliding window에 2k scores와 4k coordinates를 출력한다. 
2k scores는 해당 sliding window에 개체가 있는지 없는지를, 4k coordinates는 미리 정의된 각 앵커 박스들의 중심점 위치와 높이와 넓이를 출력한다.

![](https://velog.velcdn.com/images/seonydg/post/9bcfc12b-5a8f-4dfa-aef5-2b641c6302e4/image.png)

Faster R-CNN은 최적의 박스를 찾을 때 NMS(Non-Maximum Suppression)이라는 방법을 사용한다. 이 방법은 하나의 물체를 가리키는 여러 개의 박스들 중에서 하나만 남기는 방법으로, Confidence Score와 IOU에 의해 결정된다.
Confidence Score는 박스 안에 객체가 있을 Score의 값이다. Confidence Score를 정의하고 낮은 점수를 가진 박스들을 다 제거한다. 그리고 가장 높은 값을 가지는 박스에 대해서 주변 박스들의 IOU를 계산한다. IOU는 두 박스가 얼마나 겹쳐있는지 나타내는 값이다.(교집합/합집합)
그리고 너무 많이 겹치는 박스들을 제거하고 한 객체에 대해서 Confidence Score가 가장 높은 하나의 박스만 남게 된다. IOU Score가 일정 수준 이하로 낮다면 그것은 다른 객체로 구분하게 되어 인스턴스별 detection이 가능해진다.

Loss functions은  classification loss와 바운딩 박스 regression loss를 함계 사용한다.
그리고 평가 지표로는 mAP(mean Average Precision)을 사용한다. AP는 Precision-recall 그래프를 단조 감소 함수로 변경한 후 계산한 면적이며, 복수의 class에 대한 AP 값의 평균을 mAP라고 한다.


---
# 2. SSD
# SSD(ECCV 2016)
'SSD'는 'Faster R-CNN'의 개선 버전이라고 볼 수 있다.
Faster R-CNN에서 앵커 박스의 크기가 유연하지 못하다는 단점과 3개의 sub-network들이 유기적으로 작동하는 가에 대한 의문점이 있다.

SSD는 Feature pyramid를 사용하면 앵커 박스의 크기가 달라진다는 점에 초점을 맞춘다. 아래의 a 이미지에서 2개의 객체는 크기가 다른데, 고정된 앵커 박스의 크기가 8×8 feature maps에서는 고양이를 4×4 feature maps에서는 개를 탐지하게 된다. 즉 앵커 박스는 그대로지만 feature maps의 크기가 바뀜으로써 앵커 박스가 상대적으로 커지는 효과를 가지게 된다.

![](https://velog.velcdn.com/images/seonydg/post/a6faa27d-e738-4fc1-a0b5-0e357c42cc10/image.png)

기존부터 CNN 계열은 multi-scale-feature map을 사용해왔다. 아래의 그림과 같이 feature map 사이즈를 줄여가며 학습을 진행한 것에 착안한다.

![](https://velog.velcdn.com/images/seonydg/post/669d8e28-4512-49ca-8c69-81db4d01a3a0/image.png)

그래서 feature map이 큰 쪽에서는 작은 객체들을, 작은 쪽에서는 큰 객체들을 탐지하게 된다.
300×300 feature map을 받아서 38×38 feature map에서는 38×38×4를 그 다음 레졸루션부터는 feature map × feature map × 6 또는 4의 앵커 박스가 생기게 된다. 그래서 탐지하는 앵커 박스의 수는 8732개가 생기게 된다. 그리고 최종적으로 NMS를 적용하여 바운딩 박스를 정하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/d53425f7-bbf5-4e35-ae16-ec2ec1d29f5b/image.png)

detection model들은 포지티브 샘플(pos sample)에 비해 네거티브 샘플(neg sample)이 많을 수 밖에 없다. 그래서 학습에서 언벨런스가 생기고, 이것을 해결하기 위해 네거티브 샘플들 중에서 confidence score가 굉장히 높은 네거티브 샘플들만 학습에 사용하면서(약 1:3비율) 비율을 맞췄다.


# 3. yolov1 아키텍쳐 직접 구현
# Yolo(CVPR 2016)
yolo에서는 3단계를 통해 간단하고 실시간으로 객체 탐지를 할 수 있다.
첫 단계는 이미지를 resize하고 두 번째는 convolutional network를 통과하고 마지막으로 Non-maximum supperssion을 수행하는 것이다.

![](https://velog.velcdn.com/images/seonydg/post/00a66d85-8431-4dcb-b57c-6520c5f3e554/image.png)

yolo는 S×S로 이미지를 grid로 나누어서 하나의 grid마다 바운딩 박스를 2개씩 추론하고 해당 박스에 객체가 있는지 없는지에 대한 confidence를 같이 학습한다.
그리고 grid마다 클래스가 어떤 클래스에 속하는지 판단한다. 가장 높은 confidence score 박스에 가장 많이 포함되어 있는 class probability로 최종 클래스를 결정하게 된다.
Faster R-CNN, SSD는 백그라운드를 가리키는 바운딩 박스가 매우 많아 조절을 해야 했던 반면, yolo는 어떤 객체도 가리키지 않는 grid를 백그라운드로 인식하기에 접근 방식이 다르다.

![](https://velog.velcdn.com/images/seonydg/post/3c15b291-6808-40e2-8c1c-4148c8503a16/image.png)

feature map은 448×448로 크기로 입력이 들어오고 최종 출력은 7×7 feature resolution을 가지며, 7×7에서의 한 픽셀들이 하나의 grid가 된다.
7×7의 한 픽셀이 30개 dimension을 가지는데, 4개의 바운딩 박스 코디네이트와 해당 바운딩 박스에 대한 confidence score가 2개 존재한다. 그리고 20 classes에 대한 classification이 존재한다. 그래서 총 30개의 dimension을 가진다.

![](https://velog.velcdn.com/images/seonydg/post/53ab0f04-cca1-43d4-af5c-8fc17500ffc0/image.png)

Training Loss Funtions은 기존의 SSD, Faster R-CNN의 백그라운드 영역에 대한 바운딩 박스를 어떻게 핸들링 할 것인지가 이슈였던 반면, yolo는 객체가 있는 grid와 없는 grid로 나눠서 Loss Function을 구성한다.

![](https://velog.velcdn.com/images/seonydg/post/ea2f4d59-163d-4406-b649-4076e60277c8/image.png)

