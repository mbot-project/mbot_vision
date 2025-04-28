from ultralytics import YOLO

# Load a YOLO PyTorch model
model = YOLO("yolo11s/best.pt")

# Export the model to NCNN format
model.export(format="ncnn")