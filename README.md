# mbot_vision

## Description
A collection of computer vision examples designed for MBot.

## install
```bash
pip install ultralytics --break-system-packages
echo 'export PYTHONPATH=$PYTHONPATH:/home/mbot/.local/bin' >> ~/.bashrc
```

## Files:
- `video_streamer.py`, forward video stream to browser
- `save_image.py`, used to save image for camera calibration
- `camera_calibration.py`, standard opencv code to find camera matrix and distortion coefficients
- `apriltag_detection.py`, forward video stream to browser with apriltag detection enabled
- `apriltag_lcm_publisher.py`, publish apriltag lcm message over "MBOT_APRILTAG_ARRAY" channel
- `apriltag_lcm_subscriber.py`, subscribe to apriltag lcm message "MBOT_APRILTAG_ARRAY" channel
- `cone_detection.py`, forward video stream to browser with cone detection enabled
- `cone_lcm_publisher.py`, publish cone lcm message over "MBOT_CONE_ARRAY" channel
- `cone_lcm_publisher.py`, subscribe to cone lcm message "MBOT_CONE_ARRAY" channel
- `tag_cone_detection.py`, forward video stream to browser with apriltag and cone detection enabled

## Virtual Environment
To use this project, use venv is easier but you don't have to. But if you want to use NCNN format, then virtual env is a must.

```bash
cd ~/mbot_ws/mbot_vision
python3 -m venv mbot_vision_env --system-site-packages
source mbot_vision_env/bin/activate
python3 -m pip install --upgrade pip
```
- `flask`, this is included when we create venv with `--system-site-packages`
- `pip install ultralytics[export]` Source: [Raspberry Pi with Ultralytics YOLO11](https://docs.ultralytics.com/guides/raspberry-pi/)

## Authors and maintainers
The current maintainer of this project is Shaw Sun. Please direct all questions regarding support, contributions, and issues to the maintainer.