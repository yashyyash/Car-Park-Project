import cv2
import pickle

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('D:\MAJORPROJECTFINAL\HaarCascade\cascade11.xml')

# Load the previously defined regions of interest from the pickle files
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

try:
    with open('CarParkPosexit', 'rb') as f:
        posListExit = pickle.load(f)
except FileNotFoundError:
    posListExit = []

try:
    with open('CarParkPosentry', 'rb') as f:
        posListEntry = pickle.load(f)
except FileNotFoundError:
    posListEntry = []

# Define the width and height of each region of interest
width, height = 250, 150
width_exit, height_exit = 0, 0
width_entry, height_entry = 100, 400

# Initialize parking space counter
parking_space_counter = len(posList)

# Function to detect cars within a region of interest
def detect_cars_in_roi(img, roi):
    x, y, w, h = roi
    roi_gray = img[y:y+h, x:x+w]
    cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    return cars

# Open the camera
cap = cv2.VideoCapture(0)

# Main loop
while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reset parking space counter for each iteration
    parking_space_counter = len(posList)

    # Process each region of interest for detecting cars in parking spaces
    for pos in posList:
        x, y = pos
        roi = (x, y, width, height)
        cars = detect_cars_in_roi(gray, roi)
        if len(cars) > 0:
            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 0, 255), 2)
            cv2.putText(img, 'Car Detected', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            parking_space_counter -= 1
        else:
            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 255, 0), 2)

    # Process each region of interest for detecting cars exiting
    for pos in posListExit:
        x, y = pos
        roi = (x, y, width_exit, height_exit)
        cars = detect_cars_in_roi(gray, roi)
        if len(cars) > 0:
            cv2.rectangle(img, pos, (pos[0] + width_exit, pos[1] + height_exit), (0, 0, 255), 2)
            cv2.putText(img, 'Car is Exiting', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Process each region of interest for detecting cars entering
    for pos in posListEntry:
        x, y = pos
        roi = (x, y, width_entry, height_entry)
        cars = detect_cars_in_roi(gray, roi)
        if len(cars) > 0:
            cv2.rectangle(img, pos, (pos[0] + width_entry, pos[1] + height_entry), (0, 0, 255), 2)
            cv2.putText(img, 'Car is Entering', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the parking space counter
    cv2.putText(img, f'Parking Spaces: {parking_space_counter}', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Image", img)

    # Check for ESC key press to exit
    key = cv2.waitKey(1)
    if key == 27:  
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
