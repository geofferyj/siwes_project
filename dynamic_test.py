from keras.models import load_model              # for loading a pretrained model
from time import sleep
from keras.preprocessing.image import img_to_array  # converts images to an array of floats
import cv2      
from keras.models import model_from_json

import numpy as np     

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier = model_from_json(open("model.json", "r").read())
classifier.load_weights('mask_checker.h5')

class_labels=['The person is wearing a mask','The person is not wearing a mask']

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(70,70),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('Mask Checker',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
