import cv2
import numpy as np
import tensorflow as tf
import pickle

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('/home/yash/Car-Park-Project/HaarCascade/cascade11.xml')



# Load the previously defined regions of interest from the pickle file
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Define the width and height of each region of interest
width, height = 250, 150

# Function to detect cars within a region of interest using OpenCV
def detect_cars_in_roi(img, roi):
    x, y, w, h = roi
    roi_gray = img[y:y+h, x:x+w]
    cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    return cars

# Function to perform object detection using TensorFlow Lite
def tflite_detect_cars(frame, interpreter, min_conf=0.5, car_label_index=0):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Resize frame to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values
    input_mean = 127.5
    input_std = 127.5
    input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  

    car_detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)) and int(classes[i]) == car_label_index:
            ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
            xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
            ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
            xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))
            confidence = scores[i]  # Extract confidence score
            car_detections.append([xmin, ymin, xmax, ymax, confidence])  # Include confidence score in detection results

    return car_detections


def run_detection(camera_index, car_model_path, obj_model_path, obj_label_path):
    # Open the camera stream
    cap = cv2.VideoCapture(camera_index)

    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=obj_model_path)
    interpreter.allocate_tensors()

    # Initialize parking space counter
    parking_space_counter = len(posList)

    # Set the size of the output screen
    cv2.namedWindow('Combined Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Combined Detection', 600, 400)  # Set your desired dimensions here

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Reset the counter for each iteration
        parking_space_counter = len(posList)

        # Process each region of interest for car detection
        for pos in posList:
            x, y = pos
            roi = (x, y, width, height)
            cars = detect_cars_in_roi(frame, roi)
            if len(cars) > 0:
                cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), (0, 0, 255), 2)
                cv2.putText(frame, 'Car Detected', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                parking_space_counter -= 1
            else:
                cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), (0, 255, 0), 2)

        # Perform car detection using TensorFlow Lite
        car_detections = tflite_detect_cars(frame, interpreter)

        # Keep track of detected cars to avoid overlapping boxes
        detected_cars = []

        # Draw car detection results
        for detection in car_detections:
            # Filter out detections below 95% confidence
            if detection[4] >= 0.95:
                # Ensure there is no overlapping box
                overlap = False
                for car in detected_cars:
                    if abs(detection[0] - car[0]) < 20 and abs(detection[1] - car[1]) < 20:
                        overlap = True
                        break

                # Draw bounding box and label only if there is no overlapping box
                if not overlap:
                    cv2.rectangle(frame, (detection[0], detection[1]), (detection[2], detection[3]), (10, 255, 0), 2)
                    label = 'Car'
                    confidence = 'Confidence: {:.2f}'.format(detection[4])  # Extract confidence score from detection
                    cv2.putText(frame, label, (detection[0], detection[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)
                    cv2.putText(frame, confidence, (detection[0], detection[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)
                    detected_cars.append(detection)

        # Count the number of cars detected outside ROI
        cars_outside_roi = 0
        for detection in car_detections:
            if detection[4] >= 0.95:  # Consider only detections with confidence above 95%
                outside_roi = True
                for pos in posList:
                    if pos[0] <= detection[0] <= pos[0] + width and pos[1] <= detection[1] <= pos[1] + height:
                        outside_roi = False
                        break
                if outside_roi:
                    cars_outside_roi += 1

        # Display the parking space status and pathway occupancy
        if parking_space_counter == 0:
            cv2.putText(frame, 'Parking lot is full', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif parking_space_counter == len(posList):
            cv2.putText(frame, 'Parking lot is empty', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f'Parking Spaces: {parking_space_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the parking space status
        if cars_outside_roi <= 1:
            cv2.putText(frame, 'Pathway free to use', (int(frame.shape[1]/2) - 100, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Pathway occupied, please wait', (int(frame.shape[1]/2) - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Combined Detection', frame)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
run_detection(0, 
              r'/home/yash/Car-Park-Project/HaarCascade/cascade11.xml', 
              r'/home/yash/Car-Park-Project/Model/detect.tflite', 
              r'/home/yash/Car-Park-Project/Model/labelmap.txt')

