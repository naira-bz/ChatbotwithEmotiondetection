import cv2
import numpy as np
from tensorflow.keras.models import model_from_json  
from tensorflow.keras.preprocessing import image  
from keras.preprocessing.image import img_to_array

#load model  
classifier = model_from_json(open("model_40.json", "r").read())  

#load weights  
classifier.load_weights('model_40_weight.h5')  

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  

def EmotionRecognition():
    emotion_labels = ['neutral', 'happy','sad', 'surprise','fear', 'disgust','anger','contempt']
    count=0
    cap = cv2.VideoCapture(0)
    image = cap.read()
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
 
        cv2.imwrite("C:/Users/naira/OneDrive/Bureau/chat/env/src/ChatbotWithEmotionDetection/MedicalDiagnosis/results/emotion%d.jpg" % count, frame)  
        count=+1
          
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(50):
            break
           
    # close the camera
    #video.release()
    # close open windows
    cv2.destroyAllWindows()
    #db.session.add(frame)
    return ()
