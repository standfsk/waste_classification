# waste_classification
![image](https://github.com/cshyo1004/waste_classification/assets/60250322/9316be04-19dc-4949-9b35-e9775855d9b4)

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
