import cv2

def save_frame(camera_index=0, output_filename='captured_frame.jpg'):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return
    
    # Capture frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Couldn't capture frame.")
        cap.release()
        return
    
    # Save the frame as an image file
    cv2.imwrite(output_filename, frame)
    print(f"Frame saved as {output_filename}")

    # Release the camera
    cap.release()

# Usage
save_frame()
