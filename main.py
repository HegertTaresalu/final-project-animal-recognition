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
    camera.framerate = 24
    raw_image = PiRGBArray(camera, size(640,480))
    time.sleep(0.1)
    first_frame = None
    kernel = np.ones((20,20),np.uint8)
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        img = frame.array
        grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        grey = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        grey = cv2.medianBlur(gray,5)
        if first_frame is None:
            first_frame = grey
            raw_image.truncuate(0)
            continue
            
            absolute_difference = cv2.absdiff(first_frame,gray)
            
        _, absolute_difference = cv2.threshold(absolute_difference, 100, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) < 1:
              # Display the resulting frame
      cv2.imshow('Frame',image)
  
      # Wait for keyPress for 1 millisecond
      key = cv2.waitKey(1) & 0xFF
  
      # Clear the stream in preparation for the next frame
      raw_capture.truncate(0)
     
      # If "q" is pressed on the keyboard, 
      # exit this loop
      if key == ord("q"):
        break
     
      # Go to the top of the for loop
      continue
        
if _name__ == "__main__":
    
    #ToDo compare image values using openCV and greyscale
    main()
    
    