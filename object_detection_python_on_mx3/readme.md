# YoloV7-Tiny Demo on MemryX MX3

An example code and demo for YoloV7-Tiny running on MemryX Mx3. This repository demonstrates an end-to-end video application using the yolov7-tiny model from [here](https://github.com/WongKinYiu/yolov7) to perform object detection on a video. The 416×416 resolution model was exported to onnx and included [here](/models/yolov7-tiny_416.onnx). 

## Setup

This demo requires MemryX SDK is installed and the EVB1 (4 chips) has been properly setup:

* [SDK install](https://developer.memryx.com/docs/get_started/install.html) 
* [HW setup](https://developer.memryx.com/docs/get_started/hardware_setup.html)

This demo relies on openCV to do image capture/display.

``` bash
$ pip3 install opencv-python
```


## Run

You can run the demo using:

``` bash
$ python3 run_on_mxa.py
```

This will run yolov7-tiny inference on an included reference video. There are also some options you can specify: 

|             Option            | Description                                                     |
|-------------------------------|-----------------------------------------------------------------|
| `--video_path /path/to/video` | Path to the video to be processed. Default: "/dev/video0"       |
| `--save`                      | saves an output of the processed video to `out.avi`             |
| `--no_display`                | Optionally turn off the video display                           |


```{note} 
The inference performance might be limited by the cam FPS or the offline video file I/O.
```
