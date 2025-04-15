"""
============
Information:
============
Project: YOLOv7-tiny example code on MXA
File Name: run_on_mxa.py

============
Description:
============
A script to show how to use the Acclerator API to perform a real-time inference
on MX3 using YOLOv7-tiny model.
"""

###################################################################################################

# Imports
import time
import argparse
import numpy as np
import cv2
from queue import Queue
from threading import Thread
from matplotlib import pyplot as plt
from memryx import AsyncAccl
from yolov7 import YoloV7Tiny as YoloModel

###################################################################################################
###################################################################################################
###################################################################################################

class Yolo7Mxa:
    """
    A demo app to run YOLOv7 on the the MemryX MXA
    """

###################################################################################################
    def __init__(self, video_path, show = True, save = False):
        """
        The initialization function.
        """

        # Controls
        self.show = show
        self.save = save
        self.done = False

        # CV and Queues
        self.num_frames = 0
        self.cap_queue = Queue(maxsize=10)
        self.dets_queue = Queue(maxsize=10)
        self.vidcap = video_path
        self.dims = (640, 480) if video_path is None else ( int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
        self.color_wheel = np.array(np.random.random([20,3])*255).astype(np.int32)

        # Model
        self.model = YoloModel(stream_img_size=(self.dims[1],self.dims[0],3))

        # Timing and FPS
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros([30])

        # Vedio writer
        if save:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.writer = cv2.VideoWriter('out.avi', fourcc, self.vidcap.get(cv2.CAP_PROP_FPS), self.dims)
        else:
            self.writer = None

        # Display and Save Thread
        # Runnting the display and save as a thread enhance the pipeline performance
        # Otherwise, the display_save method can be called from the output method
        self.display_save_thread = Thread(target=self.display_save,args=(), daemon=True)
       

###################################################################################################
    def run(self):
        """
        The function that starts the inference on the MXA
        """

        # AsyncAccl
        accl = AsyncAccl(dfp='yolov7-tiny_416.dfp')

        # Start the Display/Save thread
        print("YOLOv7-Tiny inference on MX3 started")
        self.display_save_thread.start()

        start_time = time.time()

        # Gets the output from the chip and performs the cropped graph post-processing
        accl.set_postprocessing_model('yolov7-tiny_416.post.onnx', model_idx=0)

        # Connect the input and output functions and let the accl run
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()

        # Done
        self.done = True
        running_time = time.time()-start_time
        fps = self.num_frames / running_time
        print(f"Total running time {running_time:.1f}s for {self.num_frames} frames ... Average FPS: {fps:.1f}")

        # Wait for the Display/Save thread to exit
        self.display_save_thread.join()

###################################################################################################
    def capture_and_preprocess(self):
        """
        Captures a frame from the video device or Pi Camera and pre-processes it.
        """
        
        if isinstance(self.vidcap, cv2.VideoCapture):
            got_frame, frame = self.vidcap.read()
            if not got_frame:
                return None
        else:  # Pi Camera case
            frame = self.vidcap.capture_array()
            got_frame = frame is not None

        if not got_frame:
            return None

        try:
            self.num_frames += 1
            
            # Put the frame in the cap_queue to be overlayed later
            self.cap_queue.put(frame, timeout=2)
            
            # Preprocess frame
            frame = self.model.preprocess(frame)
            return frame
        
        except Queue.Full:
            print('Dropped frame .. exiting')
            return None
        
###################################################################################################
    def postprocess(self, *mxa_output):
        """
        Post process the MXA output
        """

        # Post-process the MXA ouptut
        dets = self.model.postprocess(mxa_output)

        # Push the results to the queue to be used by the display_save thread
        self.dets_queue.put(dets)

        # Calculate current FPS
        self.dt_array[self.dt_index] = time.time() - self.frame_end_time
        self.dt_index +=1
        
        if self.dt_index % 15 == 0:
            self.fps = 1 / np.average(self.dt_array)

            if self.dt_index >= 30:
                self.dt_index = 0
        
        self.frame_end_time = time.time()


###################################################################################################
    def display_save(self):
        """
        Draws boxes over the original image. It will also conditionally display/save the image.
        """

        while self.done is False:
            
            # Get the frame from and the dets from the relevant queues
            frame = self.cap_queue.get()
            dets = self.dets_queue.get()

            # Draw the OD boxes
            for d in dets:
                l,t,r,b = d['bbox']
                color = tuple([int(c) for c in self.color_wheel[d['class_idx']%20]])
                frame = cv2.rectangle(frame, (l,t), (r,b), color, 2) 
                frame = cv2.rectangle(frame, (l,t-18), (r,t), color, -1) 
                frame = cv2.putText(frame, d['class'], (l+2,t-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            if self.fps > 1:
                txt = f"{self.model.name} - {self.fps:.1f} FPS"
            else:
                txt = f"{self.model.name}"
            frame = cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2) 

            # Show the frame
            if self.show:

                cv2.imshow('YOLOv7-Tiny on MemryX MXA', frame)

                # Exit on a key press
                if cv2.waitKey(1) == ord('q'):
                    self.done = True
                    cv2.destroyAllWindows()
                    self.vidcap.release()
                    exit(1)
            
            # Save the frame
            if self.save: 
                self.writer.write(frame)

###################################################################################################
###################################################################################################
###################################################################################################

def main(args):
    """
    The main function for running YOLOv7 inference with Pi Camera
    """
    from picamera2 import Picamera2

    # Initialize and configure Pi Camera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()

    try:
        # Initialize YOLOv7 inference
        yolo7_inf = Yolo7Mxa(video_path=None, show=args.show, save=args.save)
        yolo7_inf.vidcap = picam2  # Use picam2 instead of video capture
        yolo7_inf.dims = (640, 480)  # Set dimensions to match camera config
        yolo7_inf.run()
    finally:
        picam2.stop()
        if args.show:
            cv2.destroyAllWindows()

###################################################################################################

if __name__=="__main__":
    # The args parser
    parser = argparse.ArgumentParser(description = "\033[34mMemryX YoloV7-Tiny Demo\033[0m")
    parser.add_argument('--video_path', dest="video_path", 
                        action="store", 
                        default='/dev/video0',
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam. (Default: 'samples/soccer.mp4')")
    parser.add_argument('--save', dest="save", 
                        action="store_true", 
                        default=False,
                        help="The output video will be saved at out.avi")
    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    args = parser.parse_args()

    # Call the main function
    main(args)

# eof