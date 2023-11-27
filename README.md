# object-detection

# 목록
1. Faster R-CNN
2. yolov1 아키텍쳐 직접 구현
3. data augmentation

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
