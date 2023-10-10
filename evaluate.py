
if __name__ == "__main__":
    # 라이브러리
    from ultralytics import YOLO
    import os

    # 기본 경로
    model_path = 'yolov8'
    data_path = 'data'

    # load 할 모델 명
    model_name = 'best.pt'

    # Load 모델
    model = YOLO(os.path.join(model_path, model_name))
    model.to('cuda')

    # Evaluate
    results = model.val(
        model = os.path.join(model_path, model_name),
        save_json = True,
        device = 'cuda',
        plots = True
    )
    
    # Predict
    results = model.predict(
        source = os.path.join(data_path, 'test', 'images'),
        save = True,
        device = 'cuda',
        save_conf = True,
        save_crop = True
    )