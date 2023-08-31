# pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118​
from ultralytics import YOLO
import torch

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
model.to('cuda')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='data.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg', device='cpu')

# Export the model to ONNX format
success = model.export(format='onnx')