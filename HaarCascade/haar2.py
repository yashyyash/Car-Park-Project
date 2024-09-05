import cv2
import pickle

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('D:\MAJORPROJECTFINAL\HaarCascade\cascade11.xml')

# Load the previously defined regions of interest from the pickle file
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Define the width and height of each region of interest
width, height = 150, 250

# Function to detect cars within a region of interest
def detect_cars_in_roi(img, roi):
    x, y, w, h = roi
    roi_gray = img[y:y+h, x:x+w]
    cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    return cars

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize car parking space counter
parking_space_counter = len(posList)

# Main loop
while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reset the counter for each iteration
    parking_space_counter = len(posList)

    # Process each region of interest
    for pos in posList:
        x, y = pos
        roi = (x, y, width, height)

        # Detect cars within the region of interest
        cars = detect_cars_in_roi(gray, roi)

        # If cars are detected, change the color of ROI to red and display message
        if len(cars) > 0:
            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 0, 255), 2)
            cv2.putText(img, 'Car Detected', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            parking_space_counter -= 1

        else:
            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 255, 0), 2)

    # Display the car parking space counter
    cv2.putText(img, f'Parking Spaces: {parking_space_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Image", img)

    # Check for ESC key press to exit
    key = cv2.waitKey(1)
    if key == 27:  
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
