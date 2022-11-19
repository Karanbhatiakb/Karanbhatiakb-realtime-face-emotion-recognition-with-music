
import cv2
import os
from deepface import DeepFace
import contextlib
import time

with contextlib.redirect_stdout(None): # hide pygame intro text
    from pygame import mixer 

face_cascade = cv2.CascadeClassifier('<XML file path>')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    emotion = result["dominant_emotion"]
    
    txt = str(emotion)
    if txt == "neutral":
        mixer.init() 
        mixer.music.load('songs_list/neutral.mp3') 
        mixer.music.set_volume(0.5)
        mixer.music.play(-1)
        
    elif txt =="happy":
        mixer.init()
        mixer.music.load('songs_list/happy.mp3')
        mixer.music.set_volume(0.5)
        mixer.music.play(-1)
    elif txt =="sad":
        mixer.init()
        mixer.music.load('songs_list/sad.mp3')
        mixer.music.set_volume(0.5)
        mixer.music.play(-1)
    elif txt =="angry":
        mixer.init()
        mixer.music.load('songs_list/angry.mp3')
        mixer.music.set_volume(0.5)
        mixer.music.play(-1)
    else:
        mixer.init()
        mixer.music.load('songs_list/fear.mp3')
        mixer.music.set_volume(0.5)
        mixer.music.play(-1)

    cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('frame',frame)
   
    if cv2.waitKey(100):
        time.sleep(5)
    if cv2.waitKey(500) & 0xff == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
