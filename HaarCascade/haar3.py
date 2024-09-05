import cv2
import pickle

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('D:\MAJORPROJECTFINAL\HaarCascade\cascade11.xml')

# Function to detect cars within a region of interest
def detect_cars_in_roi(img, roi):
    x, y, w, h = roi
    roi_gray = img[y:y+h, x:x+w]
    cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    return cars

# Main loop
def main():
    # Load the previously defined regions of interest for entry, exit, and parking lot from the pickle files
    try:
        with open('CarParkPosentry', 'rb') as f:
            entry_posList = pickle.load(f)
    except FileNotFoundError:
        entry_posList = []

    try:
        with open('CarParkPosexit', 'rb') as f:
            exit_posList = pickle.load(f)
    except FileNotFoundError:
        exit_posList = []

    try:
        with open('CarParkPos', 'rb') as f:
            parking_lot_posList = pickle.load(f)
    except FileNotFoundError:
        parking_lot_posList = []

    # Define the width and height of each region of interest for entry, exit, and parking lot
    entry_width, entry_height = 580, 80
    exit_width, exit_height = 580, 100
    parking_lot_width, parking_lot_height = 150, 250

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

        # Process entry region of interest
        entry_parking_space_counter = len(entry_posList)
        for pos in entry_posList:
            x, y = pos
            roi = (x, y, entry_width, entry_height)
            cars = detect_cars_in_roi(gray, roi)
            if len(cars) > 0:
                cv2.rectangle(img, pos, (pos[0] + entry_width, pos[1] + entry_height), (0, 0, 255), 2)
                cv2.putText(img, 'Car Detected', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                entry_parking_space_counter -= 1
            else:
                cv2.rectangle(img, pos, (pos[0] + entry_width, pos[1] + entry_height), (0, 255, 0), 2)

        # Process exit region of interest
        exit_parking_space_counter = len(exit_posList)
        for pos in exit_posList:
            x, y = pos
            roi = (x, y, exit_width, exit_height)
            cars = detect_cars_in_roi(gray, roi)
            if len(cars) > 0:
                cv2.rectangle(img, pos, (pos[0] + exit_width, pos[1] + exit_height), (0, 0, 255), 2)
                cv2.putText(img, 'Car Detected', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                exit_parking_space_counter -= 1
            else:
                cv2.rectangle(img, pos, (pos[0] + exit_width, pos[1] + exit_height), (0, 255, 0), 2)

        # Process parking lot region of interest
        parking_lot_parking_space_counter = len(parking_lot_posList)
        for pos in parking_lot_posList:
            x, y = pos
            roi = (x, y, parking_lot_width, parking_lot_height)
            cars = detect_cars_in_roi(gray, roi)
            if len(cars) > 0:
                cv2.rectangle(img, pos, (pos[0] + parking_lot_width, pos[1] + parking_lot_height), (0, 0, 255), 2)
                cv2.putText(img, 'Car Detected', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                parking_lot_parking_space_counter -= 1
            else:
                cv2.rectangle(img, pos, (pos[0] + parking_lot_width, pos[1] + parking_lot_height), (0, 255, 0), 2)

        # Display the car parking space counters
        cv2.putText(img, f'Entry Parking Spaces: {entry_parking_space_counter}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'Exit Parking Spaces: {exit_parking_space_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'Parking Lot Parking Spaces: {parking_lot_parking_space_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Parking Management System", img)

        # Check for ESC key press to exit
        key = cv2.waitKey(1)
        if key == 27:  
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the main loop
if __name__ == "__main__":
    main()
