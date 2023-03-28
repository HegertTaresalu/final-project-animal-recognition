"""
Loads model and tries to guess who is in the picture
image size can and should be resized based on the model loaded
The project requires Picamera, you can't replace it with usb camera
"""
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
            
"""
def main():
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    import tensorflow as tf
    import cv2
    import os
    import numpy as np#
    from keras import layers 


    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24
    raw_image = PiRGBArray(camera, size=(640,480))
    time.sleep(0.1)
    first_frame = None
    for frame in camera.capture_continuous(raw_image, format="bgr", use_video_port=True):
        kernel = np.ones((20,20),np.uint8)
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Close gaps using closing
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        # Remove salt and pepper noise with a median filter
        gray = cv2.medianBlur(gray,5)
        if first_frame is None:
            first_frame = gray
            raw_image.truncate(0)
            continue
            
        absolute_difference = cv2.absdiff(first_frame,gray)
            
        _, absolute_difference = cv2.threshold(absolute_difference, 100, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) < 1:
            print("no movement detected")
            cv2.imshow('Frame',img)
            key = cv2.waitKey(1) & 0xFF
            raw_image.truncate(0)

            if key == ord("q"):
                break
            # Go to the top of the for loop
            continue
        else:
            # Find the largest moving object in the image
            max_index = np.argmax(areas)
            # Draw the bounding box
        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        # Draw circle in the center of the bounding box
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(img,(x2,y2),4,(0,255,0),-1)
        # Print the centroid coordinates (we'll use the center of the
        # bounding box) on the image
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(img, text, (x2 - 10, y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow("Frame",img)
        # Wait for keyPress for 1 millisecond
        key = cv2.waitKey(1) & 0xFF
        # Clear the stream in preparation for the next frame
        raw_image.truncate(0)
        print("movement detected")
        # If "q" is pressed on the keyboard, 
        # exit this loop
        if key == ord("q"):
            break
if __name__ == "__main__":
    
    #ToDo compare image values using openCV and greyscale
    main()
    
    
