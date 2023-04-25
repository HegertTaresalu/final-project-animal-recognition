# final-project-animal-recognition

# What is the goal of the project?
The project's goal is to detect motion and try to identify what animal tripped the sensor.


## Requirments
* USB camera, Raspberry pi camera module or phone with droidcam(dev47apps.com) installed
* Python 3.8+
* Tensorflow 2.11(older and newer versions might work but aren't tested)
* OpenCV2
* 64 bit OS


# Setup
    The project should work with every OS on the market, but the installation is different depending on the host OS

## Mac, Linux distros(except Raspberry pi based), Windows 10 & 11:
    1.      Follow the python install guide: https://www.python.org/
    2.      pip install tensorflow==2.12.*
    3.      pip install opencv-python




## Raspberry pi

    1.      sudo apt install python3
    2.      follow the link **Insert tensorflow or tensorflow lite binary link
    3.      pip install opencv-python

The other dependencies that are required should be installed automatically by tensorflow and opencv


# Project description

The project uses Deep Learning model trained by me to identify animals that are detected by motion detection algorithm.
The model is made for image classification and not for object detection.

The project has multiple files with different algorithms. 
(Almost) Every algorithm has parameters that can be tuned to user's specific needs. 
The default parameters might works for some but its highly recommended that you try out different settings.





# Executing the project

You can execute the program by following terminal commands
        python3 main.py





# Troubleshooting



# List of operating systems the project has been tested on
* Windows 10
* Arch linux
* Raspberry PI OS

