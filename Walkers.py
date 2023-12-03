import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once the video is successfully loaded
while True:
    
    # Read the first frame
    ret, frame = cap.read()

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass the frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles drawn around detected bodies
    cv2.imshow('Body Detection', frame)

    if cv2.waitKey(1) == 32:  # 32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
