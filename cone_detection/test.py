from ultralytics import YOLO
import cv2

# Load a YOLO11n PyTorch model
# model = YOLO("yolo11n.pt")
model = YOLO("example_model.pt")

# Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
# ncnn_model = YOLO("yolo11n_ncnn_model")
ncnn_model = YOLO("example_model_ncnn_model")

# Run inference
results = ncnn_model("cone.jpg")

# Visualize the results on the frame
annotated_frame = results[0].plot()

# Save the annotated frame as an image using OpenCV
cv2.imwrite("annotated_cone.jpg", annotated_frame)