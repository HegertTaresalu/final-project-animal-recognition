import cv2
import numpy as np
import time
import tensorflow as tf
#Ignore false warning, it doesnt find import without tensorflow.keras on raspberry pi systems
import tensorflow.keras as keras
import os
import argparse
import asyncio


# Add args 
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--hide-frames", dest="show_frames", action="store_false",
                        help="Option to not show frames")
parser.add_argument("--interval", dest="interval",action="store", 
help="Integrer for how often can take images(seconds)", type = int, default = 3)
args = parser.parse_args()


# Fixed parameters for VIBE
N = 15  # Number of samples per pixel
R = 20  # Radius of the sphere
MIN_MATCHES = 2  # Number of close samples for being part of the background (bg)
PHI = 16  # Amount of random sub-sampling

# Create background and foreground identifiers
BACKGROUND = 0
FOREGROUND = 255
MIN_FOREGROUND_PIXELS = 100
# to make sure the 
MIN_TIME_BEFORE_DETECTING_MOTION = 100
# Load the model
MIN_CONTOUR_AREA = 3000

MODEL_PATH ="96%90%"
PICTURE_INTERVAL = args.interval
last_picture_time = 0
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


def detect_movement(areas,contours,frame,INTEPRETER,INPUT_DETAILS,OUTPUT_DETAILS):
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



async def calculate_segmap(segMap,samples,frame,gray_frame,height,width,rng):
    dists = np.linalg.norm(samples - gray_frame[..., np.newaxis], axis=2)
    counts = np.sum(dists < R, axis=1)
            # Classify pixel and update model
    bg_mask = counts >= MIN_MATCHES
    fg_mask = counts < MIN_MATCHES
        # Store that pixel belongs to the background
    segMap[bg_mask] = BACKGROUND
        
        # Update current pixel model
    rand_mask = (rng.integers(0, PHI, size=(height, width)) == 0)
    rand_samples = rng.integers(0, N, size=(height, width))
    samples[rand_mask, rand_samples[rand_mask]] = gray_frame[rand_mask]

    # Update neighboring pixel model
    rand_mask = (rng.integers(0, PHI, size=(height, width)) == 0)
    rand_samples = rng.integers(0, N, size=(height, width))
    x_ng = np.clip(np.random.randint(-1, 2, size=(height, width)) + np.arange(width), 0, width-1)
    y_ng = np.clip(np.random.randint(-1, 2, size=(height, width)) + np.arange(height)[:, np.newaxis], 0, height-1)
    samples[y_ng[rand_mask], x_ng[rand_mask], rand_samples[rand_mask]] = gray_frame[rand_mask]
    
    # Store that pixel belongs to the foreground
    segMap[fg_mask] = FOREGROUND

    fg = np.zeros_like(frame)
    fg[fg_mask] = frame[fg_mask]
    return segMap

async def main(show_frames=True, interval=3.0):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    start_time = time.time()
    PICTURE_INTERVAL = interval
    last_picture_time = time.time()
    ret, frame = camera.read()
    height, width, channels = frame.shape
    samples = np.zeros((height, width, N), dtype=np.uint8)
# Initialize the segmentation map

    segMap = np.zeros((height, width), dtype=np.uint8)
# Initialize the random number generator
    rng = np.random.default_rng()
    INTEPRETER, INPUT_DETAILS, OUTPUT_DETAILS = load_quantized_model(convert_model_to_quantized(load_model(MODEL_PATH)))

    while True:
        # Read the next frame from the video
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.medianBlur(frame,5)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compare pixel to background model
        segMap = await calculate_segmap(segMap,samples,frame,gray_frame,height,width,rng)

        contours, _ = cv2.findContours(segMap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        elapsed_time = time.time() - start_time

        diff_map = np.abs(gray_frame.astype(np.int32) - samples[...,-1].astype(np.int32))
        diff_map[segMap == BACKGROUND] = 0
        if elapsed_time < MIN_TIME_BEFORE_DETECTING_MOTION:
            print(f"Waiting for {MIN_TIME_BEFORE_DETECTING_MOTION - elapsed_time:.2f} seconds before detecting motion...")
            continue

            

        frame = detect_movement(areas,contours,frame,INTEPRETER,INPUT_DETAILS,OUTPUT_DETAILS)
        if time.time() - last_picture_time > PICTURE_INTERVAL:
            save_image_with_timestamp(frame, class_name)
            last_picture_time = time.time()
        if show_frames:
            cv2.imshow("diff map", diff_map.astype(np.uint8))
            cv2.imshow("frame", frame)    

        # Wait for a key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    camera.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
   asyncio.run(main(args.show_frames,args.interval))




