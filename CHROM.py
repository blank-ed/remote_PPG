# 1) Pick 500 consecutive images exhibiting the smallest amount of inter frame motion
# 2) Apply VJ face detector to define ROI
# 3) Apply simple skin selection process which produces a skin mask inside the ROI that will remove all pixels containing facial hairs and facial features
# 4) Spatial pooling of RGB
# 5) Normalize R, G and B to obtain Rn, Gn and Bn
# 6) Get Xs = 3Rn - 2Gn and Ys = 1.5Rn + Gn - 1.5Bn
# 7) Bandpass filter Xs and Ys to get Xf and Yf
# 8) Find alpha by dividing standard deviation of Xf and standard deviation of Yf
# 9) Find S = Xf - alpha*Yf
# 10)

import cv2






import cv2
import numpy as np

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("Necessary Files\\haarcascade_frontalface_default.xml")

# Start the video capture
cap = cv2.VideoCapture(r'C:\Users\Admin\Desktop\Riccardo New Dataset\test_L00_no_ex_riccardo_all_distances\D01.mp4')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply skin color detection for each face
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define a lower and upper skin color range in HSV
        lower_skin = np.array([55, 55, 55], dtype=np.uint8)
        upper_skin = np.array([255, 255, 255], dtype=np.uint8)

        # Threshold the HSV image to get only skin color
        mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)

        # Bitwise-AND the mask and original image
        roi = cv2.bitwise_and(roi, roi, mask=mask)

        # Insert the processed ROI back into the original frame
        frame[y:y+h, x:x+w] = roi

    # Display the frame with the skin mask
    cv2.imshow("Skin Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()