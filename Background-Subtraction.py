import cv2
import numpy as np
import time
import tensorflow as tf
#Ignore false warning, it doesnt find import without tensorflow.keras on raspberry pi systems
import tensorflow.keras as keras
import os
import argparse

# Add args 
parser = argparse.ArgumentParser(prog="Motion detection and image classification",
                                description="Program detects motion and tries to classify what animal tripped the sensor")

parser.add_argument("--hide-frames", dest="show_frames", action="store_false",
                        help="Option to not show frames")
parser.add_argument("--interval", dest="interval",action="store", 
help="Integrer for how often can take images(seconds)", type = int, default = 3)
args = parser.parse_args()

# Load the model
MIN_CONTOUR_AREA = 2000
MODEL_PATH ="96%90%"
IMAGE_SIZE = (180, 180)
CLASS_NAMES = np.array(["deer", "fox", "rabbit", "wild_boar"])


def load_quantized_model(QUANTIZED_MODEL):
    """Load quantized model"""
    INTEPRETER = tf.lite.Interpreter(model_content=QUANTIZED_MODEL)
    INTEPRETER.allocate_tensors()
    INPUT_DETAILS = INTEPRETER.get_input_details()
    OUTPUST_DETAILS = INTEPRETER.get_output_details()
    return INTEPRETER, INPUT_DETAILS, OUTPUST_DETAILS


def convert_model_to_quantized(MODEL):
    """ Convert the model to a quantization-aware model"""
    CONVERTER = tf.lite.TFLiteConverter.from_keras_model(MODEL)
    CONVERTER.optimizations = [tf.lite.Optimize.DEFAULT]
    QUANTIZED_MODEL  = CONVERTER.convert()
    return QUANTIZED_MODEL


def load_model(model_path):
    return keras.models.load_model(model_path)


for class_name in CLASS_NAMES:
    os.makedirs(class_name, exist_ok=True)

def save_image_with_timestamp(frame, class_name):
    """Save image with timestamp to folder named after the animal"""
    # Get the current timestamp
    timestamp = int(time.time())
    # Construct the file name
    file_name = f"{class_name}/{class_name}_{timestamp}.jpeg"
    # Save the frame as a JPEG image
    cv2.imwrite(file_name, frame)


def identify_animal(frame, x, y, w, h,INTEPRETER,INPUT_DETAILS,OUTPUST_DETAILS):
    """Identify animal in cropped image"""
    cropped_img = frame[y:y+h, x:x+w]
    cropped_img = cv2.resize(cropped_img,IMAGE_SIZE)
    # resize the cropped image to the desired size
    img_array = np.expand_dims(cropped_img, axis=0)
    img_array = img_array / 255.0
    img_array = img_array.astype("float32")
    INTEPRETER.set_tensor(INPUT_DETAILS[0]["index"], img_array)
    INTEPRETER.invoke()
    predictions = INTEPRETER.get_tensor(OUTPUST_DETAILS[0]["index"])
    # get the predicted class name and probability
    class_index = np.argmax(predictions[0])
    class_name = CLASS_NAMES[class_index]
    probability = np.round(predictions[0][class_index].astype(float) * 100, 2)
    # write the predicted class name and probability on the contour box
    text = class_name + ": " + str(probability) + "%"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
    

def augment_frame(frame,fgbg):
    frame = cv2.medianBlur(frame,5)
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
    return fgmask


def detect_movement(areas,frame,INTEPRETER,INPUT_DETAILS,OUTPUT_DETAILS):
    if len(areas) < 1:
        print("no movement detected")
        return frame
    else:
        max_contour = max(areas, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        print("movement detected")
        frame = identify_animal(frame, x, y, w, h,INTEPRETER, INPUT_DETAILS, OUTPUT_DETAILS)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame,(x2,y2),4,(0,255,0),-1)
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

def main(show_frames=True, interval = 3.0):
    """Main function that initiates camera and applies image augmentation to frames"""
    PICTURE_INTERVAL = interval
    last_picture_time = time.time()
    camera = cv2.VideoCapture(0)
    time.sleep(0.1)
    INTEPRETER, INPUT_DETAILS, OUTPUT_DETAILS = load_quantized_model(convert_model_to_quantized(load_model(MODEL_PATH)))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=350, varThreshold=25, detectShadows=False)    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame")
            break
        fgmask = augment_frame(frame,fgbg)
        # Get the contours and their areas
        contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # Filter out small contours
        areas = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]        
        frame = detect_movement(areas,frame,INTEPRETER,INPUT_DETAILS,OUTPUT_DETAILS)
        if show_frames:
            cv2.imshow("Frame",frame)
            cv2.imshow("fgmask",fgmask)

        if time.time() - last_picture_time > PICTURE_INTERVAL:
            save_image_with_timestamp(frame, class_name)
            last_picture_time = time.time()

        if cv2.waitKey(1) == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(args.show_frames,args.interval)

