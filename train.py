
if __name__ == "__main__":
    # 라이브러리
    from ultralytics import YOLO
    import os

    # 기본 경로
    model_path = 'yolov8'
    data_path = 'data'

    # 새로운 YOLO model 생성
    model_name = YOLO('yolov8n.yaml')

    # Load 모델
    model = YOLO(os.path.join(model_path, model_name))
    model.to('cuda')

    # Train
    results = model.train(
        model = os.path.join(model_path, model_name),
        data = os.path.join(model_path, 'data.yaml'),
        epochs = 10,
        patience = 5,
        batch = 16,
        seed = 42,
        optimizer = 'AdamW',
        pretrained = True,
        device = 'cuda',
        save = True,
        save_period = 1,
        amp=False,
        lr0=0.001
    )