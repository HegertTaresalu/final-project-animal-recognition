from pyimagesearch.tempimage import TempImage
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import cv2
import os
from keras import layers
from keras.utils import to_categorical



camera = PiCamera()
camera.resolution = (1024, 768)






def identifyImage():
    model = keras.models.load_model("model_name")
    image_size = (180,180)
    class_names = ["deer", "fox", "rabbit", "wild_boar"]
    img = keras.preprocessing.image.load_img(
    f, target_size=image_size)
    predictions = model.predict(img_array)
    numbers = predictions[0]
    for i in range(len(class_names)):
        print(class_names[i] ," : ", round(numbers[i].astype(float)* 100, 2) , "%")
            
