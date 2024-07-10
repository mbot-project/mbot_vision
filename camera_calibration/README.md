# Camera Calibration

## Description
Camera calibration scripts for MBot use.

## Installation
To use this tool, you will need Flask, it might come with the system or you might have to install it. To check if you have flask:
```bash
python -c "import flask"
```

## Usage and Features
To find the intrinsic matrix of the mbot camera, we need to perform the camera calibratoin.

1. Find a Checkerboard

    Obtain a checkerboard from the lab. The size of the checkerboard is determined by the number of intersections, not squares. For instance, a notation of [8, 6] means the board has 8 intersections along one dimension and 6 along the other, corresponding to a grid of 9 by 7 squares.

2. Collect Images

    Execute the following command and navigate to `http://your_mbot_ip:5001` to gather calibration images:
    ```bash
    # if you use nvidia jetson nano
    $ python3 save_image_jetson.py
    # if you use pi 5
    $ python3 save_image_pi5.py
    ``` 
    - This will save the images in the `/images` directory for camera calibration.
    - Aim to capture images from various angles and distances for optimal results.

3. Start Calibration

    Adjust the code in `camera_calibration.py` to match your checkerboard's specifications:
    ```python
    # TODO: 
    # 1. Adjust the CHECKERBOARD dimensions according to your checkerboard
    # 2. Set the correct square_size value
    CHECKERBOARD = (6, 8)    
    square_size = 25    # in millimeters
    ```
    Then run the following command:
    ```bash
    $ python3 camera_calibration.py 
    ```
    - This uses the images in `/images` to generate the calibration data `cam_calibration_data.npz`.
    - Uncomment the print statements in the code if you wish to view the calibration results.

4. Verify Calibration Results
    - A "Mean reprojection error" below 0.5 indicates successful calibration, and the camera matrix is reliable.
    - If the "Mean reprojection error" is significantly above 0.5, verify the accuracy of your CHECKERBOARD dimensions and square_size settings.


## Authors and maintainers
The current maintainer of this project is Shaw Sun. Please direct all questions regarding support, contributions, and issues to the maintainer. 