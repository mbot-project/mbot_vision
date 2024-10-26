# Cone Detection

Source: [Raspberry Pi with Ultralytics YOLO11](https://docs.ultralytics.com/guides/raspberry-pi/)

We follow the guide with the "without docker" approach:
```bash
cd ~/mbot_ws/mbot_vision
python3 -m venv yolo_env --system-site-packages
source yolo_env/bin/activate
```
```bash
pip install -U pip
pip install ultralytics[export]
```