import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
#import tensorflow as tf
#import tensorflow.keras as keras

"""
def identifyImage():
    model = keras.models.load_model("96%90%")
    image_size = (180,180)
    class_names = ["deer","fox", "rabbit", "wild_boar"]
    img = keras.preprocessing.image.load_img("test.jpeg", target_size=image_size)
    predictions = model.predict(img_array)
    numbers = predictions[0]
    for i in range(len(class_names)):
        print(class_names[i], " : " , round(numbers[i].astype(float)* 100, 2),"%")
"""
def main(use_usb_camera=False):
    if use_usb_camera:
        camera = cv2.VideoCapture(0)
    else:
        identifyImage()
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 24
    raw_image = PiRGBArray(camera, size=(640,480))
    time.sleep(0.1)
    prev_gray = None
    for frame in camera.capture_continuous(raw_image, format="bgr", use_video_port=True):
        kernel = np.ones((20,20),np.uint8)
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Close gaps using closing
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        # Remove salt and pepper noise with a median filter
        gray = cv2.medianBlur(gray,5)

        if prev_gray is not None:
            # Calculate the optical flow between the previous frame and the current frame
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Calculate the magnitude and direction of the flow vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Set the mask to display only the motion that is above a certain threshold
            mask = np.zeros_like(img)
            mask[..., 1] = 255
            mask[..., 0] = np.where(magnitude > 30, 255, 0)

            # Find contours in the mask image
            contours, _ = cv2.findContours(mask[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw a bounding box around each contour
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Frame', img)

        prev_gray = gray

        # Wait for keyPress for 1 millisecond
        key = cv2.waitKey(1) & 0xFF
        # Clear the stream in preparation for the next frame
        raw_image.truncate(0)

        if key == ord("q"):
            break

    if use_usb_camera:
        camera.release()
    else:
        camera.close()

if __name__ == "__main__":
    main()
