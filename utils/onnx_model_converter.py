from ultralytics import YOLO

# Load a YOLO PyTorch model
model = YOLO("yolov8n/best_v8n.pt")

# Export the model to NCNN format
model.export(format="onnx") 