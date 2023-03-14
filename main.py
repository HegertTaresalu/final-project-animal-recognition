"""
Loads model and tries to guess who is in the picture
image size can and should be resized based on the model loaded
The project requires Picamera, you can't replace it with usb camera
"""


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
            

def main():
    from pyimagesearch.tempimage import TempImage
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import tensorflow as tf
    import cv2
    import os
    from keras import layers
    from keras.utils import to_categorical


    #Setup camera	
    camera = PiCamera()
    camera.resolution = (640, 480)
    raw_image = PiRGBArray


if _name__ == "__main__":
    
    #ToDo compare image values using openCV and greyscale
    main()
    
    