
# Cone Detection Starter Code and Examples

## 0. Dependencies
```bash
python3 -m venv cone_env --system-site-packages
pip install ultralytics==8.0.200
pip install pandas
pip install flask
pip install cvzone
```


## 1. Required LCM Changes
   We must create a new LCM message for publishing/subscribing. <br />
   1. mbot_ws/mbot_lcm_base/mbot_lcm_serial/lcm_config.h<br />
      a. Add: #define MBOT_CONE_ARRAY_CHANNEL "MBOT_CONE_ARRAY"
   2. mbot_ws/mbot_lcm_base/mbot_msgs/lcmtypes<br />
      a. Create: mbot_cone_array_t.lcm<br />
         i. package mbot_lcm_msgs;

            struct mbot_cone_array_t
            {
               int64_t utime;
               int32_t array_size;
               mbot_cone_t detections[array_size];
            }

      b. Create: mbot_cone_t.lcm<br />
         i. package mbot_lcm_msgs;

            struct mbot_cone_t
            {
               string color;
               float range;
               float heading;
            }

   3. mbot_ws/mbot_lcm_base/mbot_msgs/CMakeLists.txt<br />
      a. To set(LCM_FILES ...) Add: lcmtypes/mbot_cone_t.lcm and lcmtypes/mbot_cone_array_t.lcm

## 2. Colored Cone Detection
   Description: Here we train a custom Yolo-V8 model for detecting colored cones (this will theoretically work with any objects you want to detect!). We provide a starter model for you, but the model could be improved by training on more images (current model is trained on ~20 images).<br />
   Needs: Raspberry Pi or Jetson, Calibrated and Functioning MBot Camera, Google Colab, Python<br />

   1. Start by using the MBot camera to collect several pictures of the cones you want to detect in the environment you wish to detect them in. This should include at least 20 pictures, but this will work better with more pictures (~50). These pictures should include different cominations of cone poses and lighting conditions. See example photos below:<br />
      <insert example pictures><br />
   2. Create the dataset using your desired annotation tool. Here, we use Roboflow, but other options will work as well, such as LabelImg, V7, and LabelMe.<br />
      a. If this is your first time using RoboFlow, create a free account<br />
      b. Select "Create New Project"<br />ors and maintainers
The original author of this project is Cameron Harris.ing". If you wish to split the work among teammates, you can invite them to the project at this stage and assign images to label to different members.<br />
      f. Once the annotation tool pops up, use the polygonal tool to outline the cones. In this tutorial, we divide the cones into classes by color, however you could assign all cones to the same class if the color does not matter. With the desired cone outlined, type "<color>_cone" in the "Label" dialogue box. Each time you create a new label class, RoboFlow will save it so you can classify other cones of that color with the same label.<br />
      ![image](https://github.com/camharris99/mbot_cv/assets/122319358/e428483d-c5ca-4472-8494-da2458040325)<br />

      g. Repeat Step f for each cone in the image, for each image in the dataset.<br />
      h. Once all images are annotated, under "Annotation", select "Add <#> Images to Dataset".<br />
      i. Under "Generate", you can select some features to speed up training/add extra images without taking more pictures (augmentation). For this model, we used Pre-Processing steps of: Auto-Orient and Resize to 640 x 640. We augmented the dataset with an Image Level Blur. Once desired settings are chosen, hit "Create".<br />
      j. Under "Versions", hit "Export Dataset", and download a zip file to your computer using the YoloV8 settings.<br />
      ![image](https://github.com/camharris99/mbot_cv/assets/122319358/dc8305b5-7795-42ac-ae1f-d2d4adea7234)<br />
   3. Verify your dataset is in the proper format for YoloV8.<br />
      a. You should have a data.yaml file with the folder locations of Train, Test, and Validation, and the list of model Classes ("red_cone", "blue_cone", etc).<br />
      b. There should also be 3 folders (Train, Test, Valid), each containing 2 subfolders: Images and Labels.<br />
   4. Upload your dataset zip folder to your Google Drive.<br />
   5. Open the provided Jupyter Notebook in Google Colab and complete all of the cells. Don't forget the final step of downloading best.pt after training!


## Authors and maintainers
The original author of this project is Cameron Harris.