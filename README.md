# 🚗 Fatigue driving detection system based on yolov7 #

Introduction:Based on the yolov7 and openseeface open source library, we implemented the detection of fatigue driving behaviour in the target video, in which our detection targets mainly include open mouth, closed eyes, looking left and right, and so on.


## Implementation framework ##
In the implementation framework, we frame draw the obtained video images, for the obtained video frames are checked for anomalies, and if there are more than three consecutive seconds of anomalous behaviours, they are judged as fatigue driving behaviours.
![image](https://github.com/Cadezzyr/Fatigued-driving-based-on-yolov7/blob/main/video/frame1.png)

## Demo ##
### Look left and right ###
![image](https://github.com/Cadezzyr/Fatigued-driving-based-on-yolov7/blob/main/video/tinywow_video_2_63218709.gif)
### Close eyes ###
![image](https://github.com/Cadezzyr/Fatigued-driving-based-on-yolov7/blob/main/video/tinywow_vidieo_1_63218660.gif)
### Open mouth ###
![image](https://github.com/Cadezzyr/Fatigued-driving-based-on-yolov7/blob/main/video/tinywow_video_3_63218792.gif)
