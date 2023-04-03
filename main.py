import cv2
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras


model = keras.models.load_model("96%90%")

def identifyAnimal(frame, x, y, w, h):
    image_size = (180, 180)
    class_names = ["deer","fox", "rabbit", "wild_boar"]
    # crop the image to the bounding box of the moving object
    cropped_img = frame[y:y+h, x:x+w]
    img_array = keras.preprocessing.image.img_to_array(cropped_img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.image.resize(img_array, size=image_size)
    # get predictions from the model
    predictions = model.predict(img_array)
    # get the predicted class name and probability
    class_index = np.argmax(predictions[0])
    class_name = class_names[class_index]
    probability = round(predictions[0][class_index].astype(float) * 100, 2)
    # write the predicted class name and probability on the contour box
    text = class_name + ": " + str(probability) + "%"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame



def main(use_usb_camera=True):
    if use_usb_camera:
        camera = cv2.VideoCapture(0)
    else:
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 24
    time.sleep(0.1)
    first_frame = None
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame")
            break
        kernel = np.ones((20,20),np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Close gaps using closing
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        # Remove salt and pepper noise with a median filter
        gray = cv2.medianBlur(gray,5)
        if first_frame is None:
            first_frame = gray
            continue

        absolute_difference = cv2.absdiff(first_frame,gray)

        _, absolute_difference = cv2.threshold(absolute_difference, 100, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) < 1:
            print("no movement detected")
            cv2.imshow('Frame',frame)
            key = cv2.waitKey(1) & 0xFF

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
            frame = identifyAnimal(frame, x, y, w, h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            # Draw circle in the center of the bounding box
            x2 = x + int(w/2)
            y2 = y + int(h/2)
            cv2.circle(frame,(x2,y2),4,(0,255,0),-1)
            # Print the centroid coordinates (we'll use the center of the
            # bounding box) on the image
            text = "x: " + str(x2) + ", y: " + str(y2)
            cv2.putText(frame, text, (x2 - 10, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Display the resulting frame
            cv2.imshow("Frame",frame)
            # Wait for keyPress for 1 millisecond
            key = cv2.waitKey(1) & 0xFF

            print("movement detected")
            cv2.imwrite("test.jpeg", frame)
            # If "q" is pressed on the keyboard,
            # exit this loop
            if key == ord("q"):
                break
    if use_usb_camera:
        camera.release()
    else:
        camera.close()


if __name__ == "__main__":
    main()

