from memryx import NeuralCompiler
nc = NeuralCompiler(
    num_chips=2,  # Adjust based on your hardware (e.g., MX3)
    models="yolov8n/best_v8n.onnx",
    dfp_fname="yolo8n",
    autocrop=True  # Splits postprocessing for host execution
)
dfp = nc.run()