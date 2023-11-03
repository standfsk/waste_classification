# 쓰레기 분류 프로젝트
![image](https://github.com/cshyo1004/waste_classification/assets/60250322/9316be04-19dc-4949-9b35-e9775855d9b4)

## 프로젝트 개요
재활용품 분리배출을 실천하기 가장 힘든 이유는 분리배출 방법이 다양하고 많기 때문에 정확하게 알아보기 귀찮다는 것이 가장 큰 이유다. 

## 목표
사용자들이 다양하고 복잡한 쓰레기 분리배출 방법을 보다 편리하게 이해하고 실천할 수 있도록 도와주는 모델 개발을 목표로 한다. 

## How to Install
```
$ git clone https://github.com/cshyo1004/waste_classification.git)https://github.com/cshyo1004/waste_classification.git
$ pip install -r requirements.txt
```

## Data
<a href="https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=140">AIHUB 생활폐기물 이미지 데이터셋</a>

## Model
<a href="https://github.com/ultralytics/ultralytics">Yolov8</a>

## Performance
|  | train1 | train2 | train3 | train4 |
| --- | --- | --- | --- | --- |
| epochs | 10 | 10 | 10 | 10 |
| patience | 5 | 5 | 5 | 5 |
| batch_size | 16 | 16 | 32 | 32 |
| optimizer | AdamW | AdamW | AdamW | AdamW |
| learning_rate | 0.01 | 0.001 | 0.01 | 0.001 |
| box loss | 0.31641 | 0.31976 | 0.32979 | 0.31192 |
| cls loss | 0.25473 | 0.2663 | 0.3003 | 0.25236 |
| 학습시간 | 93h | 95h | 93h | 95h |

#### box_loss
![image](https://github.com/cshyo1004/waste_classification/assets/60250322/4f82145c-7510-4f42-99ee-d92cfc922e7a)
#### cls_loss
![image](https://github.com/cshyo1004/waste_classification/assets/60250322/bb541fef-7390-46c0-b9f6-80a0fdaecd0f)
