from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

output= {
    'Happy':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\happy.jpg",
    'Angry':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\angry.jpg",
    'Sad':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\sad.jpg",
    'Neutral':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\sad.jpg",
    'Surprise':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\sad.jpg",
    # 'Sad':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\sad.jpg",
    # 'Sad':r"C:\Users\satya\OneDrive\Documents\codechef_contest\questions\face-expression-recognition-dataset\face-expression-recognition-dataset\OutPut empji\sad.jpg"
}


face_classifier = cv2.CascadeClassifier(r'C:\Users\satya\OneDrive\Documents\codechef_contest\questions\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\satya\OneDrive\Documents\codechef_contest\questions\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(640,480),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    img1=frame
    img2=cv2.resize(cv2.imread(output['Angry']),(20,480))

    # cv2.imshow('Emotion Detector',frame)
    Hori = np.concatenate((img1, img2), axis=1)
    img2=cv2.resize(cv2.imread(output['Angry']),(640,480))
    Verti = np.concatenate((img1, img2), axis=0)
 
    cv2.imshow('HORIZONTAL', Hori)
    cv2.imshow('VERTICAL', Verti)
    # print(label)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























