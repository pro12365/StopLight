import cv2

# Load the pre-trained traffic light detection model
model = cv2.CascadeClassifier('light.xml')

# Define the colors for traffic lights
color_dict = {0: 'Red', 1: 'Yellow', 2: 'Green'}

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Loop through each frame in the video stream
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect traffic lights in the grayscale frame
    lights = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # Loop through each detected traffic light
    for (x, y, w, h) in lights:
        # Crop the traffic light region from the frame
        roi = frame[y:y+h, x:x+w]

        # Resize the traffic light region to a fixed size
        roi = cv2.resize(roi, (32, 32))

        # Convert the resized region to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculate the mean color of the HSV region
        mean = cv2.mean(hsv_roi)

        # Determine the color of the traffic light based on the mean color
        if mean[0] > 150:
            color = 2 # Green
        elif mean[0] > 80:
            color = 1 # Yellow
        else:
            color = 0 # Red

        # Draw a rectangle around the detected traffic light and label it with the color
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, color_dict[color], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the processed frame with detected traffic lights
    cv2.imshow('Traffic Light Detection', frame)

    # Exit the program if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()