# object-detection

# 목록
1. Faster R-CNN
2. SSD
3. yolov1 아키텍쳐 직접 구현
4. yolov1 아키텍쳐 data augmentation
5. EfficientDet
6. Swin Transformer
7. Finding Tiny Faces

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


# 5. EfficientDet
# EfficientDet

여러 아키텍쳐들의 방법들을 사람이 아닌 자동으로 찾는 AI를 개발하는 제안을 하게 된다.
그래서 NAS(Neural Architecture Search)라는 연구 분야가 발생하게 된다. 

![](https://velog.velcdn.com/images/seonydg/post/c584003e-e5ab-45e7-9972-61c380ce15f2/image.png)

NAS를 간단히 보면, 먼저 사람이 정해놓은 Search Space가 있고 Search Space를 탐색할 Search Strategy를 정한다. 그 후에 Search Strategy 통해서 찾은 아키텍쳐를 찾고 performance를 측정한다. 그리고 performance를 리워드로 사용하거나 이 performance를 기준으로 다른 Search Strategy에서 새로운 아키텍쳐를 찾아내는 방법을 반복한다.

NAS는 노말셀과 리덕션셀 두 가지를 찾는데, 너무 복잡하게 되어 있었다. 커널 사이즈부터 오퍼레이션까지 모두 다르고 복잡하게 연결이 되다보니 성능은 좋지만 실사용에서 적용시키기에 의문점이 많이 있었다. 

이 후에는 얼마나 Efficient하게 그리고 아키텍쳐 상에서 채널의 수나 블록의 반복 횟수들에 대한 방향으로 가게 된다. 그래서 NAS의 큰 틀은 비슷하나 새로운 아키텍쳐를 찾는다기 보다는 최적화된 아키텍쳐를 기계가 자동으로 찾아가는 방향 접근하게 된다.

그렇게 나온 논문이 아래 그림의 EfficientNet(CVPR 2019)이다.

![](https://velog.velcdn.com/images/seonydg/post/a1dafb60-26a0-48b5-91e6-6a3019095fb5/image.png)


채널 수, 레이어, 입력 영상의 레졸루션 등을 scaling하면서 찾아보겠다는 것이다. 리소스가 굉장이 많이 드는 연구로 일반적으로 접근하기에는 어려움이 있다. 

**EfficientDet**은 EfficientNet을 detection용으로 바꿔서 제안한 방법이다. 기존의 state of the Art Method보다 좋은 성능을 보여준다. 

![](https://velog.velcdn.com/images/seonydg/post/8cbf63da-5f63-4b21-b46b-c7544f0bf5f6/image.png)

EfficientDet은 크게 3가지를 제안한다.
- Cross-scale connections
- Weighted feature fusion
- Compound scaling

### 1. Cross-scale connections

![](https://velog.velcdn.com/images/seonydg/post/f44c3b46-5415-42f0-9632-46a29ac2f7df/image.png)

FPN에서 P의 숫자가 클 수록 피쳐 레졸루션이 작은 상태인데, 레졸루션을 줄여나가다가 다시 레졸루션을 키우면서 이전의 피쳐들에 맞게 스케일링 혹은 더하는 방법으로 합친다. 그리고 다시 출력을 통해서 최종 detection 성능을 뽑는 방법으로 사용된다. U-Net과 비슷하지만, U-Net은 다시 P0까지 업스케일링 후에 출력을 한 것에 반해서 FPN은 원하는 결과 출력부에 Head를 연결시켜서 출력을 하게 된다.
PANet은 FPN의 다운과 업을 반복하는 방법을 제안했다.
NAS-FPN은 백본 이후의 네트워크를 NAS 방법을 통해 찾는 방법으로 복잡한 형태의 커넥션을 만든다.
BiFPN은 기존의 FPN과 같이 레졸루션의 연결이 올라갔다가 내려오는 구조는 유지하고 기존의 레졸루션에서 스킵 커넥션을 가져오는 방식을 취한다.

### 2. Weighted feature fusion

![](https://velog.velcdn.com/images/seonydg/post/8fd605ae-d19c-41aa-bb30-21ac69920f5f/image.png)

각각의 features에 가중치를 줘서 합한다. unbounded fusion은 각 피쳐 맵마다 어떤 가중치를 정해줘서 최종 출력을 정하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/d6bedf89-9f32-4011-8553-78be5a3f95cb/image.png)

EfficientDet 논문에서는 unbounded fusion보다 softmax-based fusion을 하는 것이 좋다고 제안한다. 하지만 softmax-based fusion은 계산량이 많이 들어가게 된다.

![](https://velog.velcdn.com/images/seonydg/post/5f5ff3b1-65f6-4460-a21d-1a54b35c4661/image.png)


그래서 비슷한 형태를 가지지만 e의 지수승을 계산하기 보다 하나의 변수로 변경을 한 fast normalized fusion을 제안한다.

예로 아래의 P6의 td.

![](https://velog.velcdn.com/images/seonydg/post/b324cac8-6bfb-4344-8c67-b060169bc71d/image.png)


### 3. Compound scaling

![](https://velog.velcdn.com/images/seonydg/post/98a41891-b69c-4007-ae0b-bd5fbba6a6d1/image.png)

논문에서는 스케일링 팩터 파이를 정한 다음에, 그 파이에 따라서 반복 횟수, 영상 레졸루션 등을 정하게 된다.
W(bifpn)는 채널수, D(bifpn)는 BiFPN 레이어를 몇 번 반복할지 정의, D(box, class)는 네트워크에 있는 conv의 수, R(input)은 입력으로 파이가 증가함에 따라서 선형적으로 증가하도록 정의가 되어 있다.

아래와 같이 표로 정리해 볼 수 있다.
이렇게 미리 정의함으로써 네트워크가 심플해진다.

![](https://velog.velcdn.com/images/seonydg/post/64f35233-804d-451d-a47a-31cb91617b02/image.png)

![](https://velog.velcdn.com/images/seonydg/post/7f292fd7-0b76-4aa2-8ccc-977887b36e42/image.png)


### Loss Function

Loss Function은 RetinaNet에서 사용한 Focal Loss를 사용한다. 
Pt는 score는 에측하지 못한 0과 잘 예측한 값 1사이의 값을 가지는데, CE(cross entropy loss)가 0이면 inf loss를 1이면 loss를 받지 않는다. CE는 잘못 예측한 경우에 패털티를 부여하는데 초점을 맞춘 것이다. 그래서 샘플 수가 다른 것에 대해서도 모두 같은 loss값을 주기때문에 누적이 되면 샘플 수가 많은(백그라운드) 것에 예측을 잘 하지만 정답에 대해서는 예측을 잘 하지 못하는 경우가 발생한다.
Focal Loss는 예측된 확륙값을 기반으로 Loss를 조절하는 방법이다. 기존의 CE에 **'(1 - pt)^gamma'**을 곱해준다. 그래서 gamma가 0이면 CE와 같고 커질수록 x축과 가까워지는 형태가 된다.

![image](https://github.com/seonydg/object-detection/assets/85072322/58f13024-4460-4c76-b2a0-a7ef62b845ef)


---
# 6. Swin Transformer
# Swin Transformer
Swin Transformer를 보기에 앞서 Transformer가 무엇인지 간단히 보도록 하자.


## Transformer(NIPS 2017)

Transformer는 input과 output 사이의 컴포넌트들의 집합으로 표현이 된다. 

![](https://velog.velcdn.com/images/seonydg/post/8cf27da7-0ceb-4312-b828-1e4bb669bc76/image.png)

Transformer의 컴포넌트들은 Encoder 혹은 Encoder와 Decoder들로 이루어져 있다. Encoder와 Decoder로 이루어진 Transformer를 주력으로 보자면, Target이 있는 경우는 원하는 출력 형태가 있을 때 유용하게 사용이 된다.(ex. 번역)

![](https://velog.velcdn.com/images/seonydg/post/f10a5564-8263-4f27-8abc-3406eabe437c/image.png)

Encoder와 Decoder 블럭의 상세 정보는 다음과 같다.
입력값에 Positional Endocing이 들어가는데, NLP는 시퀀스의 순서 또는 이미지의 경우는 grid상의 위치 정보가 표기되지 않기 때문에 위치 정보를 따로 넣어준다.

![](https://velog.velcdn.com/images/seonydg/post/81e21b19-7bae-4761-a106-58f28ffc6675/image.png)

input이 들어오게 되면(X) key, query, value 프로젝션 메트릭스를 곱해서 백터로 만든다. Q와 K를 MatMul을 통해서 score를 구한 다음에 스케일링을 하고 Softmax를 통과하고 V와 다시 MatMul을 한다.

![](https://velog.velcdn.com/images/seonydg/post/ca0920f1-346e-411c-8267-7d14088c2132/image.png)

좀 더 자세히 보자면,
입력값(X)에 Q, K, V를 곱해서 Q, K, V 백터를 생성한다. 

![](https://velog.velcdn.com/images/seonydg/post/434cbef3-975d-4c0f-83dc-331a0b958e84/image.png)

그 다음 Q, K 매칭을 통해 score를 구한다. 지금의 예는 self-attention이기에 같은 문장에서의 백터들을 곱한다. 그리고 그림에서의 행별로 Softmax를 취해 각각 확률값들을 만들어준다.

![](https://velog.velcdn.com/images/seonydg/post/7b36a0fb-c0df-43d8-9aa4-0f8c7cf32a93/image.png)

각각 만들어진 attention score들을 V와 MatMul을 통해서 output를 만든다.
이 과정을 attention 블럭을 병렬적으로 사용한다.

![](https://velog.velcdn.com/images/seonydg/post/2985e68c-58e2-4631-8146-98fefdd12f01/image.png)

self-attention을 거친 백터와 거치지 않은 백터를 더해서 Normalization을 해준 후 FFN로 넘긴다. 보통 FFN에서는 MLP(Multi-Layer Perceptron, 그림에선 2 layers를 거치는)를 사용한다. 그리고 다시 FFN을 거친 것과 거치지 않은 벡터들을 합치고 Normalization을 하여 내보낸다.

잠시 self-attention에 의미를 보자.
self-attention는 의미를 찾기 위한 장치다. 의미는 사물 간 관계의 결과이고, self-attention은 관계를 배우는 일반적인 방법, 즉 우리가 어디에 주의를 기울여야 하는지 학습하는 것이다.
ex)
1) 그는 주전자의 물을 컵에 따랐다. 그것이 가득 찰 때까지. -> 그것 : 컵
2) 그는 주전자의 물을 컵에 따랐다. 그것이 텅 빌 때까지. -> 그것 : 주전자

![](https://velog.velcdn.com/images/seonydg/post/5935e337-e7cb-4670-95c8-43af4451774e/image.png)

Decoder 부분도 self-attention과 동일한 방법으로 진행한다. 주의점은 Encoder와 Decoder에서 파라미터들은 서로 다른 파라미터들이다.

![](https://velog.velcdn.com/images/seonydg/post/4b833e97-9af6-4413-847a-c2ae8005b8b0/image.png)

다른 점은 Enc-Dec Attention에서 K, V는 Encoder에서 넘어온 백터를 사용한다는 것이다.
그리고 Encoder에서의 FFN과 같은 방법을 거쳐서, 테스크에 따라서 이미지 객체 탐지라면 바운딩 박스를, 번역이라면 단어들에 대한 확률값 등을 거쳐 최종 output를 내보내게 된다.


### Transformer 간단 Code

```
def multi_head_attention(Q, K, V):

    num_batch, num_head, num_token_length, att_dim = K.shape
    Q = Q / (att_dim**0.5)

    attention_score = Q @ K.permute(0,1,3,2) # num_batch, num_head, num_token_length, num_token_length

    attention_score = torch.softmax(attention_score, dim=3)

    Z = attention_score @ V # num_batch, num_head, num_token_length, att_dim

    return Z, attention_score


class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        
        num_batch, num_head, num_token_length, att_dim = K.shape
        Q = Q / (att_dim**0.5)

        attention_score = Q @ K.permute(0,1,3,2) # num_batch, num_head, num_token_length, num_token_length

        attention_score = torch.softmax(attention_score, dim=3)

        Z = attention_score @ V # num_batch, num_head, num_token_length, att_dim

        return Z, attention_score


class EncoderLayer(torch.nn.Module):

    def __init__(self, hidden_dim, num_head, dropout_p=0.5):
        super().__init__()

        self.num_head = num_head
        self.hidden_dim = hidden_dim

        self.MHA = MultiHeadAttention()

        self.W_Q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_O = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.LayerNorm1 = torch.nn.LayerNorm(hidden_dim)
        self.LayerNorm2 = torch.nn.LayerNorm(hidden_dim)

        self.Dropout = torch.nn.Dropout(p=dropout_p)

        self.Linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.Activation = torch.nn.ReLU()


    def to_multihead(self, vector):
        num_batch, num_token_length, hidden_dim = vector.shape
        att_dim = hidden_dim // self.num_head
        vector = vector.view(num_batch, num_token_length, self.num_head, att_dim) # [num_batch, num_token_length, num_head, att_dim]
        vector = vector.permute(0,2,1,3) # [num_batch, num_head, num_token_length, att_dim]
        return vector


    def forward(self, input_Q, input_K, input_V):
        # input_Q : [num_batch, num_token_length, hidden_dim]

        Q = self.W_Q(input_Q) # [num_batch, num_token_length, hidden_dim]
        K = self.W_K(input_K)
        V = self.W_V(input_V)

        num_batch, num_token_length, hidden_dim = Q.shape

        Q = self.to_multihead(Q) # [num_batch, num_head, num_token_length, att_dim]
        K = self.to_multihead(K)
        V = self.to_multihead(V)


        Z , _ = self.MHA(Q,K,V) # [num_batch, num_head, num_token_length, att_dim]
        Z = Z.permute(0,2,1,3)  # [num_batch, num_token_length, num_head, att_dim]
        Z = Z.reshape(num_batch, num_token_length, self.hidden_dim) # [num_batch, num_token_length, hidden_dim]

        Z = self.W_O(Z)

        Z = self.LayerNorm1( self.Activation(Z) + input_Q)
        Z1 = self.Dropout(Z)

        Z = self.Activation(self.Linear1(Z1))
        Z = self.Dropout(Z)
        Z = self.Activation(self.Linear2(Z))
        Z = self.Dropout(Z)

        Z = Z + Z1

        Z = self.LayerNorm2(Z)

        return Z
```

## Swin Transformer

Transformer를 기반으로 비젼 문제를 푸는 모델이다.
아래는 ImageNet Classification들의 결과인데, ViT(Vision Transformer) 기반의 알고리즘들이 연구되고 발표되면서 비전에서 큰 각광을 받게 된다.
단점은 Transformer는 엄청난 양의 데이터와 계산을 필요로하며 시간도 오래 걸리는 테스크들이 많다. 그래서 큰 데이터와 자본을 가지고 있는 메이져 기업이 아니면 진행하기 힘든 연구라는 것이다.
그래서 메이저 기업들에서 사전 학습된 모델의 backbone을 올려주고, 이것을 파인 튜닝해서 연구를 진행하는 형식으로 진행이 되고 있다.

![](https://velog.velcdn.com/images/seonydg/post/3d3fa07b-a304-4a45-81f0-2b5641e8be03/image.png)


### ViT

ViT를 간단히 말해보자면, 기존 NLP에서 글자로 되어 있던 부분을 이미지 패치로 바꾸어 적용시켰다고 볼 수 있다. 아래의 그림과 같이 영상이 주어져 있으면, 영상을 패치 단위로 쪼개고 Linear Projection을 통해서 백터화 시킨다. 그리고 이미지 패치에 따라서 백터가 주어지고 그것을 Encoder에 넣는 방식이다.

![](https://velog.velcdn.com/images/seonydg/post/9467815b-3fbf-4811-8a97-1794f1f8adb9/image.png)

미리 정해진 방법대로 영상의 패치를 쪼개서 각 패치들끼리의 관계를 attention을 통해 찾는 것이 ViT였다. 
반면 Swin Transformer는 패치를 매우 작게 쪼갠 것과 더 크게 키운 패치들을 한 번에 볼 수 있는 Hierarchical feature map을 만든다. 그래서 객체가 작고 큰 경우, 영상 전체 maps를 보고 분류를 할 것인지 작은 부분을 보고 분류를 할 것인지 판단하게 된다. 다양한 패치들의 크기를 가지고 다양하게 비전 테스크를 수행한다.

![](https://velog.velcdn.com/images/seonydg/post/29a4e36f-433b-4347-87ac-b3c3ceafa071/image.png)

Swin Transformer Shifted window 방법을 사용한다.
아래 빨간색의 테두리가 하나의 패치인데, 이 패치를 다시 쪼개서 패치 안에서 더 작게 쪼갠 패치들의 attention score를 계산한다. 그리고 빨간색 패치의 위치를 바꾸어서 다시 진행하게 된다. 레이어별로 이동하면서 attention을 계산할 window를 이동시키는 것이다.

![](https://velog.velcdn.com/images/seonydg/post/089902ae-9717-4526-b760-d2420ae084ae/image.png)

Swin Transformer의 아키텍쳐를 보면, 가장 작은 패치 단위를 1/4×1/4로 큰 패치를 1/32×1/32로 잡았다. 
입력에서 1/4×1/4로 쪼갰기에 디멘전을 맞추기 위해서 3에 16을 곱해서 48로 맞추어 입력을 시킨다. 그리고 C(channel)로 바꾸고 패치를 키우게 되면서 채널도 2배씩 키우면서 디멘전을 맞추어 나간다. 이 방법은 기존 CNN에서 피쳐 레졸루션을 줄여가면서 채널 수를 늘려가는 방식과 같다고 볼 수 있다.
LN은 layers normalization, W-MSA는 Window Multi-Head Self-Attention, SW-MSA는 Shifted-Window Multi-Head Self-Attention방법으로, 전체 구성은 ViT와 비슷하게 되어 있다.

![](https://velog.velcdn.com/images/seonydg/post/a713a14d-2949-4572-80ac-da67d21d63c6/image.png)

---
# 7. Finding Tiny Faces
# Finding Tiny Faces(CVPR 2017)

아주 작은 얼굴들을 탐지하기 위해서 어떤 기술적 테크닉들이 필요한지 살펴보자.

![](https://velog.velcdn.com/images/seonydg/post/5bea9c94-b147-49f9-a964-4535a18c740f/image.png)

객체 탐지를 위해 5가지의 approaches를 이야기 한다.

먼저 고정된 사이즈의 얼굴을 디텍션할 수 있는 템플릿을 정해놓고 이미지 피라미드를 만들어서 해당 얼굴 영역을 찾는다. 원본 상태에서 영상 크기를 줄여가며 여러 템플릿에 탐지를 하여 구체적인 얼굴이 매칭되는 것을 찾을 수 있도록 한다.
그리고 기존의 SSD나 YOLO처럼 이미지 안의 여러 스캐일별 얼굴을 탐지할 수 있는 방법으로 찾는다.
나머지 3가지 테크닉을 제시한다.
먼저 이미지 피라미드를 만들고 이미지의 스케일과 템플릿(앵커 박스)의 크기를 복합해서 쓰며, 얼굴만이 아니라 fixed-size의 얼굴을 포함하는 것을 보고 분류를 해야한다고 제시한다. 마지막으로 마지막 레이어의 피쳐만 사용하는 것이 아니라 앞쪽 레이어의 리쳐들도 함께 사용해야 함을 제시한다.

![](https://velog.velcdn.com/images/seonydg/post/dbd90ec9-f21f-4b19-86c3-ca0d287f789e/image.png)

![](https://velog.velcdn.com/images/seonydg/post/26b28efb-bbb0-4afd-b1d7-c670eff7c6b3/image.png)


기존의 SSD나 YOLO 및 ResNet 등에서 사용되던 방법들이다. 그래서 무엇이 다른가.

### Context information to find small faces
아래의 그림에서 왼쪽 위의 그림은 작은 얼굴을 아래는 큰 얼굴을 갖는 그림이다. 그리고 초록색은 작은 바운딩 박스를 파란색은 얼굴에 맞는 세배 비율의 박스를 노란색은 fixed-size로 얼굴에 상관없이 300필셀로 구분을 한다.

아래 오른쪽 그림은 얼굴 사이즈가 다를 때의 결과인데, context를 반영한 것이 결과가 더 좋다는 것을 반영하고 있는 그림이다. 이 성능의 차이가 사이즈가 작아질수록 fixed-size가 훨씬 좋음을 알 수 있다.

![](https://velog.velcdn.com/images/seonydg/post/d8f1dfc5-8cca-4b85-99da-e4be901ef49c/image.png)

아래 그림에서, face size를 두고 receptive field가 커질수록 정확도가 높아지는 경향이 있음을 알 수 있다.

![](https://velog.velcdn.com/images/seonydg/post/94212531-806e-4000-99be-4c704fd09e4a/image.png)



### Foveal descriptor using multiple layers in a deep network
하나의 피쳐만 사용하는 것이 아니라, ResNet skip-connector처럼 앞쪽 레이어의 피쳐를 더했을 때 성능이 더 좋아지는 것을 확인할 수 있다. 특히 **res3-4**까지는 비슷하나 **res5**까지 가서 비교를 했을 때 확연한 차이가 있고 작은 얼굴을 탐지할 때 월등한 성능 향상이 있음을 확인할 수 있다. 그래서 논문에서는 Context information을 위해서 무조건 마지막 레이어에서만 결과를 출력하는 것은 반드시 좋은 결과를 나타내지 않는다고 주장한다.

![](https://velog.velcdn.com/images/seonydg/post/01fa93ab-7e57-4deb-9e62-a9b454d09222/image.png)



### Image scale
얼굴의 크기에 따라서 객체 탐지의 성능이 달라지는데, 그렇다면 얼굴을 탐지하기 위해서 가장 좋은 패치 사이즈는 얼마인지를 고민하게 된다. 

아래 오른쪽 그림에서 작은 얼굴은 사이즈를 2배 키웠을 때 약 10% 정도의 성능 향상을 가지고, 큰 얼굴은 사이즈를 2배 줄였을 때 약 5% 정도의 성능 향상을 가지고 있다. 즉 무조건 얼굴 사이즈가 크다고 성능 향상과 직결되지는 않는다고 주장한다. 그래서 논문 저자들은 객체의 크기보다 크기들의 분포에 이슈가 있다고 말한다. 

아래 왼쪽 그림에서처럼, 너무 작은 사이즈나 큰 사이즈는 샘플 수가 부족해서 탐지가 안되는 것이라 주장을 한다. 

![](https://velog.velcdn.com/images/seonydg/post/437dfe2e-5646-4aa4-8ae5-7a40c7eb99ea/image.png)


그렇다면 어떻게 사이즈가 다른 객체들을 크기에 상관없이 잘 탐지할 수 있을까.
논문에서는, CNN 모델을 특정한 객체 사이즈에 맞춰 학습하는 방법을 제시하게 된다. 

입력 영상이 주어졌을 때, 기존 영상과 2배 키운 영상과 2배 줄인 영상인 3가지의 피라미드를 만들고 end-to-end로 학습을 진행한다. 그리고 기존의 영상에서는 중간 정도의 객체를 2배 키운 영상에서는 작은 객체를 반으로 줄인 영상에서는 큰 객체를 추출을 한다. 즉 바운딩 박스의 필셀 크기로 로스를 정의하고 탐지를 한다. 그리고 추출된 객체들을 합치고 NMS를 통해 최종 출력을 하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/6efc6af9-793f-41ca-976c-d6e6cc31f90e/image.png)


결과를 보면 반으로 줄인 사이즈에서는 큰 얼굴의 탐지가 잘 되고, 2배로 키운 사이즈는 큰 얼굴은 탐지가 잘 되지 않고 작은 얼굴일수록 탐지가 잘 되는 것을 볼 수 있다. 
영상의 사이즈별로 탐지를 잘하는 영역이 다르기에, 잘 하는 영역에 맞게 최종 출력물의 모델을 다르게 써서 문제를 해결한다. 그래서 모든 사이즈들을 모두 합쳐서 출력을 내면 Full Model이고, 2가지를 합친 Model이나 하나의 Model도 사용할 수 있다. 아래의 오른쪽 그림은 그것에 대한 결과물이다. 데이터셋과 그 결과물을 보면서 해당 데이터셋에 맞는 모델을 선정하면 될 것이다.


![](https://velog.velcdn.com/images/seonydg/post/72ea79ea-eb34-4c44-9cbd-53aca21ac595/image.png)
