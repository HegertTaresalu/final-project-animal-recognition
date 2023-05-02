import cv2
import numpy as np
import time
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import tensorflow.keras as keras
import os
import argparse
# Add args 
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--hide-frames", dest="show_frames", action="store_false",
                        help="Option to not show frames")
parser.add_argument("--interval", dest="interval",action="store", 
help="Integrer for how often can take images(seconds)", type = int, default = 3)
args = parser.parse_args()

# Load the model
model = keras.models.load_model("96%90%")

# Convert the model to a quantization-aware model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Load the quantized model
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

picture_interval = args.interval
last_picture_time = 0
image_size = (180, 180)
class_names = np.array(["deer", "fox", "rabbit", "wild_boar"])

for class_name in class_names:
    os.makedirs(class_name, exist_ok=True)

def save_image_with_timestamp(frame, class_name):
    """Save image with timestamp to folder named after the animal"""
    # Get the current timestamp
    timestamp = int(time.time())
    # Construct the file name
    file_name = f"{class_name}/{class_name}_{timestamp}.jpeg"
    # Save the frame as a JPEG image
    cv2.imwrite(file_name, frame)


def identify_Animal(frame, x, y, w, h):
    """Identify animal in cropped image"""
    cropped_img = frame[y:y+h, x:x+w]
    cropped_img = cv2.resize(cropped_img,image_size)
    # resize the cropped image to the desired size
    img_array = np.expand_dims(cropped_img, axis=0)
    img_array = img_array / 255.0
    img_array = img_array.astype("float32")
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    # get the predicted class name and probability
    class_index = np.argmax(predictions[0])
    class_name = class_names[class_index]
    probability = np.round(predictions[0][class_index].astype(float) * 100, 2)
    # write the predicted class name and probability on the contour box
    text = class_name + ": " + str(probability) + "%"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
    

def main(show_frames=True, interval = 3.0):
    """Main function that initiates camera and applies image augmentation to frames"""
    global last_picture_time
    picture_interval = interval
    last_picture_time = time.time()
    camera = cv2.VideoCapture(1)
    time.sleep(0.1)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame")
            break
        fgmask = fgbg.apply(frame)
        kernel = np.ones((20,20),np.uint8)
        gray = fgmask
        # Close gaps using closing
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        # Remove salt and pepper noise with a median filter
        gray = cv2.medianBlur(gray,5)

        # Get the contours and their areas
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("no movement detected")
            continue
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        print("movement detected")
        frame = identify_Animal(frame, x, y, w, h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame,(x2,y2),4,(0,255,0),-1)
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if show_frames:
            cv2.imshow("Frame",frame)

        if time.time() - last_picture_time > picture_interval:
            save_image_with_timestamp(frame, class_name)
            last_picture_time = time.time()
        if cv2.waitKey(1) == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(args.show_frames,args.interval)

