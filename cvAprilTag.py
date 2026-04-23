import numpy as np
import cv2 as cv
import pupil_apriltags as apriltag

cap = cv.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create an AprilTag detector
detector = apriltag.Detector()

# Loop to continuously get frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to get a frame")
        break
    
    # Convert the frame to grayscale
    grayScaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect AprilTags in the grayscale image
    tags = detector.detect(grayScaled)
    # TODO Figure out camera parameters
    #tags = detector.detect(grayScaled, estimate_tag_pose=True, camera_params=[600, 600, 320, 240], tag_size=0.1)

    for tag in tags:
        # Get the corners of the detected tag
        corners = tag.corners.astype(int)
        
        # Draw a bounding box around the tag
        cv.polylines(grayScaled, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw the tag ID
        cv.putText(grayScaled, str(tag.tag_id), (corners[0][0], corners[0][1] - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print the tag information to the console
        # print(tag)

        # TODO: Pose_t & Pose_R can be used to get the position and orientation of the tag in 3D space
    
    # Display the frame with detected tags
    cv.imshow('Frame', grayScaled)
    
    # Exit on 'q' key press
    if cv.waitKey(1) == ord('q') or cv.getWindowProperty('Frame', cv.WND_PROP_VISIBLE) < 1:
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()