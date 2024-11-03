# mbot_vision

## Description
A collection of computer vision examples designed for MBot

## Virtual Environment
To use this project, you have to use python virtual enviroment.

```bash
cd ~/mbot_ws/mbot_vision
python3 -m venv mbot_vision_env --system-site-packages
source mbot_vision_env/bin/activate
python3 -m pip install --upgrade pip
pip install -e .
```

### Check cone detection message use mbot lcm-spy
```bash
mbot lcm-spy --channels MBOT_CONE_ARRAY --module mbot_vision_lcm
```
### Check apriltag detection message use mbot lcm-spy
```bash
mbot lcm-spy --channels MBOT_APRILTAG_ARRAY
```

### Uninstall
```bash
pip uninstall mbot_vision
```
### Get out of virtual env
```bash
deactivate
```

## Generate LCM message
```bash
lcm-gen -p mbot_cone_* --ppath ..
```

## Dependencies
- `flask`, this is included when we create venv with `--system-site-packages`
- `pip install ultralytics[export]` Source: [Raspberry Pi with Ultralytics YOLO11](https://docs.ultralytics.com/guides/raspberry-pi/)

## Authors and maintainers
The current maintainer of this project is Shaw Sun. Please direct all questions regarding support, contributions, and issues to the maintainer.