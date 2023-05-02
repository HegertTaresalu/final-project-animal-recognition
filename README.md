# final-project-ML

# What is the goal of the project?
The project's goal is to detect motion and try to identify what animal tripped the sensor.


## Requirments
* USB camera, Raspberry pi camera module or phone with droidcam(dev47apps.com) installed
* Python 3.8+
* Tensorflow 2.11(older and newer versions might work but aren't tested)
* OpenCV2
* Numpy


# Setup
   The project works on System that support OpenCV and Tensorflow 2.12


## Mac, Linux distros(except Raspberry pi based), Windows 10 & 11:
    1. Follow the python install guide: https://www.python.org/
    2. pip install tensorflow==2.11.*
    3. pip install opencv-python




## Raspberry pi
    1. sudo apt install python3
    2. follow the link https://github.com/PINTO0309/Tensorflow-bin and install using the provided guide
    3. pip install opencv-python

The other dependencies that are required should be installed automatically by tensorflow and opencv


# Project description

The project uses Deep Learning model trained by me to identify animals that are detected by motion detection algorithm.
The model is made for image classification and not for object detection.
Also i provided pretrained image classification model incase mine is not suitable for the required job.

The project has multiple files with different algorithms. 
(Almost) Every algorithm has parameters that can be tuned to user's specific needs. 
The default parameters might works for some but its highly recommended that you try out different settings.





# Executing the project
You can execute the program by following terminal command

    python3 main.py [--hide-frames][--interval]






 # Troubleshooting
 
 ## Error quantizing model

 If you get error that contains something along: 
``` 
error endvector missing 1 required positional
``` 
Error is caused by Tensorflow and Flatbuffers mismatch 
Try to update the packages by pip, if its unable to find any newer version then download and install it manually using pypi
Following installs
```
pip3 install flatbuffers-23.3.3-py2.py3-none-any.whl
```


## Out of Index Error
To troubleshoot an "out of index" error caused by the `camera` variable, ensure that the camera is properly connected. Try changing the argument passed to `cv2.VideoCapture()` to `0` if you are using the default camera, or try different integer values if you have multiple cameras connected. Also, check that the `camera.read()` method is returning a valid frame before attempting to access it. You can also add print statements to check the values of `ret` and `frame` before using them. 




# List of operating systems the project has been tested on
* Windows 10
* Arch linux
* Raspberry PI OS

