import cv2
import numpy as np 
from keras.models import model_from_json

# Dictionary to map predicted labels to emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON model
json_file = open("C:\\Users\\User\\Desktop\\pfe\\code\\emotio_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights("C:\\Users\\User\\Desktop\\pfe\\code\\emotio_model.h5")
print("Model loaded successfully")

# Initialize video capture
use_webcam = True  # Set this to False to use an image file
if use_webcam:
    cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it to use a different camera
else:
    image_path = "C:\\Users\\User\\Desktop\\pfe\\code\\IMG-20220516-WA0021.jpg"  # Change this to the path of your image file
    frame = cv2.imread(image_path)

while True:
    if use_webcam:
        # Read frame from the camera
        ret, frame = cap.read()

        # Check if frame is read successfully
        if not ret:
            print("Error: Failed to capture frame")
            break
    else:
        use_webcam = True  # Only use the image file once
        cap.release()  # Release the webcam capture resource

    # Resize the frame
    frame = cv2.resize(frame, (720, 720))

    # Detect faces in the frame
    face_detector = cv2.CascadeClassifier('C:\\Users\\User\\Desktop\\pfe\\code\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_face = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in num_face:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), axis=0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)

    # Display the frame with emotions detected
    cv2.imshow("Emotion Detection", frame)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture resource and close all OpenCV windows
if use_webcam:
    cap.release()
cv2.destroyAllWindows()
