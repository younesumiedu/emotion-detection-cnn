import cv2
import numpy as np 
from keras.models import model_from_json

emotio_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Suprised"}

#load json model
json_file = open("C:\\Users\\User\\Desktop\\pfe\\code\\emotio_model.json",'r')#C:\\Users\\SOULAIMAN BOUALI\\emotio_model.json
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

#load weights into new model 
emotion_model.load_weights("C:\\Users\\User\\Desktop\\pfe\\code\\emotio_model.h5")
print("loaded of model is sucefully")

    #si on veut utiliser le camera
    #cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture("C:\\Users\\User\\Desktop\\pfe\\code\\IMG-20220516-WA0021.jpg")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(720,720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('C:\\Users\\User\\Desktop\\pfe\\code\\haarcascade_frontalface_default.xml')
    gray_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_face = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_face:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img =  np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1),axis=0)

        #predict the emotion
        #cropped_img = cropped_img.reshape(-1, 48, 48, 1)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotio_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
    cv2.imshow("emotion detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
while(True):    
    cap.release(1)
    cv2.waitKey(0)
#cv2.destroyAllWindows()