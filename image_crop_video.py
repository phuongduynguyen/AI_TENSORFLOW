import re
import numpy as np
import cv2
import json
import time
import os

path="cropped"
count=1
#load haar cascade classifier to crop face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#load video
cap = cv2.VideoCapture('VID20220106220005.mp4')
#loop through the video
while(cap.isOpened()):
    #read frame
    ret,frame=cap.read()
    count=count+1
    #after every 5 frame save the face
    #can change this number to get different face expression
    if (count%5==0):
        
        print(count)
        #if frame is not null
        if ret==True:
            #convert image from BRG to Gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #pass it through face classifier
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            #loop through each faces in frame
            for (x,y,w,h) in faces:
                #crop face frome image
                roi_color = frame[y:y+h, x:x+w]
                #save face in jpg format
                cv2.imwrite(path+"/"+str(count)+".jpg",roi_color)
        #if frame is null
        else:
            #release loaded video
             cap.release()
