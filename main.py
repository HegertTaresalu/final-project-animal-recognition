import cv2
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
import os
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
image_size = (180, 180)
class_names = np.array(["deer", "fox", "rabbit", "wild_boar"])

picture_interval = 3.0


for class_name in class_names:
    os.makedirs(class_name, exist_ok=True)

def save_image_with_timestamp(frame, class_name):
    # Get the current timestamp
    timestamp = int(time.time())
    # Construct the file name
    file_name = f"{class_name}/{class_name}_{timestamp}.jpeg"
    # Save the frame as a JPEG image
    cv2.imwrite(file_name, frame)


def identifyAnimal(frame, x, y, w, h):
    cropped_img = frame[y:y+h, x:x+w]
    cropped_img = cv2.resize(cropped_img,image_size)
    # resize the cropped image to the desired size
    img_array = np.expand_dims(cropped_img, axis=0)
    img_array = img_array / 255.0
    img_array = img_array.astype("float32")
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    # get the predicted class name and probability
    class_index = np.argmax(predictions[0])
    class_name = class_names[class_index]
    probability = np.round(predictions[0][class_index].astype(float) * 100, 2)
    # write the predicted class name and probability on the contour box
    text = class_name + ": " + str(probability) + "%"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    global last_picture_time
    if time.time() - last_picture_time > picture_interval:
        save_image_with_timestamp(frame, class_name)
        last_picture_time = time.time()
    return frame
def main():
    global last_picture_time
    last_picture_time = time.time()
    camera = cv2.VideoCapture(0)
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
        frame = identifyAnimal(frame, x, y, w, h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame,(x2,y2),4,(0,255,0),-1)
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Frame",frame)
        print("movement detected")
        if time.time() - last_picture_time > picture_interval:
            save_image_with_timestamp(frame, class_name)
            last_picture_time = time.time()
    camera.release()


if __name__ == "__main__":
	main()

