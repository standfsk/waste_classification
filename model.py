if __name__ == "__main__":

    from ultralytics import YOLO
    import torch
    import sys
    import os
    sys.path.append('D:/waste_datasets')
    model_path = 'yolov8'
    data_path = 'data'

    try:
        # Create a new YOLO model from scratch
        # model = YOLO('yolov8n.yaml')

        # Load a pretrained YOLO model (recommended for training)
        model = YOLO(os.path.join(model_path, 'yolov8x.pt'))
        model.to('cuda')

        # Train the model
        results = model.train(
            model = os.path.join(model_path, 'yolov8x.pt'),
            data = os.path.join(model_path, 'data.yaml'),
            epochs = 100,
            patience = 5,
            batch = 32,
            seed = 42,
            optimizer = 'AdamW',
            pretrained = True,
            device = 'cuda',
            save = True,
            save_period = 1
        )

        # Evaluate the model's performance on the validation set
        results = model.val(
            model = os.path.join(model_path, 'yolov8x.pt'),
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

    except Exception as e:
        print(e)

    # Export the model to ONNX format
    # success = model.export(format='onnx', path='.')
