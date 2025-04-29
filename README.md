# mbot_vision

## Description
A collection of computer vision examples designed for MBot. This is a toolbox, picl what you need. All the files under the root directory is a standalone program.

## install
Follow instructions [here](https://rob550-docs.github.io/docs/botlab/how-to-guide/mbot-vision-guide.html).

## Virtual Environment
To use this project, use venv is easier but you don't have to.

```bash
cd ~/mbot_ws/mbot_vision
python3 -m venv mbot_vision_env --system-site-packages
source mbot_vision_env/bin/activate
python3 -m pip install --upgrade pip
pip install ultralytics
pip install --no-cache-dir "ncnn"
```

```bash
# if use logger
pip install matplotlib
pip install pandas
pip install seaborn
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
- `tag_cone_lcm_publisher.py`, publish cone lcm message over "MBOT_CONE_ARRAY" channel, and publish apriltag lcm message over "MBOT_APRILTAG_ARRAY" channel
- `tag_cone_lcm_subscriber.py`, as name stated, subscribe to both of the detections
- `cone_detection_train.ipynb`, cone detection training notebook, details see comments there
- `ncnn_model_converter.py` convert the model format to [ncnn](https://docs.ultralytics.com/integrations/ncnn/)


## How to create system service for teleop
```bash
chmod +x controller_teleop.py
sudo cp ~/mbot_ws/mbot_vision/services/mbot_teleop.service /etc/systemd/system/ 
sudo systemctl daemon-reload 
sudo systemctl enable mbot_teleop.service
sudo systemctl start mbot_teleop.service
```

## Authors and maintainers
The current maintainer of this project is Shaw Sun. Please direct all questions regarding support, contributions, and issues to the maintainer.