import numpy as np
import cv2 as cv
import pupil_apriltags as apriltag

cap = cv.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Loop to continuously get frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to get a frame")
        break
    
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('Frame', gray)

    # TODO AprilTag detection
    
    # Exit on 'q' key press
    if cv.waitKey(1) == ord('q') or cv.getWindowProperty('Frame', cv.WND_PROP_VISIBLE) < 1:
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()