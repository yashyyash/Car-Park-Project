import cv2
import pickle

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('D:\MAJORPROJECTFINAL\HaarCascade\cascade11.xml')

# Load the previously defined regions of interest from the pickle file
try:
    with open('CarParkPosentry', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Define the width and height of each region of interest
width, height = 80, 580

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

    # Process each region of interest
    for pos in posList:
        x, y = pos
        roi = (x, y, width, height)

        # Detect cars within the region of interest
        cars = detect_cars_in_roi(gray, roi)

        # Change color based on car detection
        color = (0, 255, 0)  # Green by default
        if len(cars) > 0:
            color = (0, 0, 255)  # Red if car detected

        # Draw rectangle representing parking space with appropriate color
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)

        # Display a message if a car is detected entering
        if len(cars) > 0:
            cv2.putText(img, 'Car is Entering', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Image", img)

    # Check for ESC key press to exit
    key = cv2.waitKey(1)
    if key == 27:  
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
