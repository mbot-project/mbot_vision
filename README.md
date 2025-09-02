# mbot_vision

## Description
A collection of computer vision examples designed for MBot. This is a toolbox, pick what you need. All the files under the root directory is a standalone program.

## Use Case 1 - Classroom
Follow ROB 550 instructions [here](https://rob550-docs.github.io/docs/botlab/how-to-guide/mbot-vision-guide.html).


## Use Case 2 - MBot follower demo
Start from a fresh mbot classic base image.
### Install
Install LCM base
```bash
cd ~
mkdir mbot_ws
cd ~/mbot_ws
git clone https://github.com/mbot-project/mbot_lcm_base
cd ~/mbot_ws/mbot_lcm_base
./scripts/install.sh
```
Install apriltag
```bash
cd ~
git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF
sudo cmake --build build --target install
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.11/site-packages' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```

Cloning mbot_vision repository
```bash
cd ~/mbot_ws
git clone https://github.com/mbot-project/mbot_vision.git
```
### Test the camera, calibrate, and test apriltag
> You don't have to do this for the leader.

Following the commands in the ROB550 instruction [here](https://rob550-docs.github.io/docs/botlab/how-to-guide/mbot-vision-guide.html).
1. First test the camera using the command under "Testing the Setup"
2. Then calibrate the camera following "Camera Calibration" section
3. Finally run the following command to visualize the tag detection
    ```bash
    python3 apriltag_detection.py
    ```

### Create system service
On the leader mbot:
```bash
cd ~/mbot_ws/mbot_vision
chmod +x controller_teleop.py
sudo cp ~/mbot_ws/mbot_vision/services/mbot-teleop.service /etc/systemd/system/ 
sudo systemctl daemon-reload 
sudo systemctl enable mbot-teleop.service
sudo systemctl start mbot-teleop.service
```

On all the follower mbot:
```bash
cd ~/mbot_ws/mbot_vision
chmod +x apriltag_follower.py
sudo cp ~/mbot_ws/mbot_vision/services/mbot-follower.service /etc/systemd/system/ 
sudo systemctl daemon-reload 
sudo systemctl enable mbot-follower.service
sudo systemctl start mbot-follower.service
```

### Final result
Once both the leader and followers have booted up, you can control the leader using the controller, and the followers will automatically begin tracking the nearest AprilTag without any additional configuration.

Control scheme:
  - Left stick Up/Down: Forward/Backward movement
  - Right stick Left/Right: Turning
  - Left shoulder buttons (L1/L2): Increase/decrease max linear speed
  - Right shoulder buttons (R1/R2): Increase/decrease max angular speed
  - Default max linear speed: 0.10 m/s
  - Default max angular speed: 0.50 rad/s

If the leader robot doesn't move:
1. The controller is not ON
2. The control board lost connection, press the RST button on it to reboot the control board
## Use Case 3 - Development
### Virtual Environment
To use this project, use venv is recommended
```bash
cd ~/mbot_ws/mbot_vision
python3 -m venv mbot_vision_env --system-site-packages
source mbot_vision_env/bin/activate
python3 -m pip install --upgrade pip
pip install ultralytics
pip install --no-cache-dir "ncnn"
```

This is legacy code for cone detection data analysis:
```bash
# if use logger
pip install matplotlib
pip install pandas
pip install seaborn
```

## Files
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


## Authors and maintainers
The current maintainer of this project is Shaw Sun. Please direct all questions regarding support, contributions, and issues to the maintainer.