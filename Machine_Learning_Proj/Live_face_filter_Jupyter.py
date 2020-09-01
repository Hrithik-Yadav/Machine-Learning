# -*- encoding used: utf-8 -*-
"""
Created on Sat Jul 11 22:54:17 2020

author -: Hrithik Yadav
"""

import cv2
import matplotlib.pyplot as plt
import csv
import os

#We first load the XML pre-trained classifiers data.
cascade_path = "F:\\hrithik's resume\\Machine_Learning_Proj\\Haarcascade_proj\\haarcascades\\"
data_path = "F:\\hrithik's resume\\Machine_Learning_Proj\\Haarcascade_proj\\Data\\"

face_cascade_eye_right = cv2.CascadeClassifier(cascade_path + 'haarcascade_righteye_2splits.xml')
face_cascade_eye_left = cv2.CascadeClassifier(cascade_path + 'haarcascade_lefteye_2splits.xml')
face_cascade_nose = cv2.CascadeClassifier(cascade_path + 'nose.xml')


#these functions are used to draw the mustache and  and glasses on our image
def draw_glasses(img1,bound_eye):
    if(len(bound_eye)>=2):
        #img = cv2.resize(img,(600,600))
        try:
            x_o = int(bound_eye[0][0] - 25)
            y_o = int(bound_eye[0][1] - 25)
            x_o = abs(x_o)
            y_o = abs(y_o)
            x_r = 2*((bound_eye[0][0]+bound_eye[1][0]+bound_eye[1][2])/2) - 2*x_o - 10
            y_r = max(bound_eye[1][3],bound_eye[0][3])+75
            x_r = abs(x_r)
            y_r = abs(y_r)
            img2 = cv2.imread(data_path + 'Train\\glasses.png',-1)
            img2 = cv2.resize(img2,(int(x_r),int(y_r)))
            
            y1, y2 = y_o, y_o + img2.shape[0]
            x1, x2 = x_o, x_o + img2.shape[1]
        
            alpha_s = img2[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
        
            for c in range(0, 3):
                try:
                    img1[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] + alpha_l * img1[y1:y2, x1:x2, c])
                except():
                    pass
        except():
            pass
    return img1


def draw_mustache(img1,bound_nose):
    if(len(bound_nose)>=1):
        #img = cv2.resize(img,(600,600))
        try:
            x_o = int(bound_nose[0][0] - (bound_nose[0][3]/bound_nose[0][2])*5)
            y_o = int(bound_nose[0][1]+(bound_nose[0][3])/2 + 8)
            x_o = abs(x_o)
            y_o = abs(y_o)
            x_r = 1.5*(bound_nose[0][2])
            y_r = max(bound_nose[0][2]/2,0)
            x_r = abs(x_r)
            y_r = abs(y_r)
            img2 = cv2.imread(data_path + 'Train\\mustache_edited.png',-1)
            img2 = cv2.resize(img2,(int(x_r),int(y_r)))
            
            y1, y2 = y_o, y_o + img2.shape[0]
            x1, x2 = x_o, x_o + img2.shape[1]
        
            alpha_s = img2[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
        
            for c in range(0, 3):
                try:
                    img1[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] + alpha_l * img1[y1:y2, x1:x2, c])
                except():
                    pass
        except():
            pass
    return img1


def image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img1):
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #The cascade classifiers work only on grayscale colors-space.
    eye_r = face_cascade_eye_right.detectMultiScale(gray, 1.3, 5)    
    eye_l = face_cascade_eye_left.detectMultiScale(gray, 1.3, 5)
    nose = face_cascade_nose.detectMultiScale(gray, 1.3, 5)
    bound_eye = []  #storing these outputs as a list of tuples from the detect multiscale function. 
    bound_nose = []
    for (x,y,w,h) in eye_r:
        #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2) #these are used to draw rectangle on our image
        bound_eye.append([x,y,w,h])
    for (x,y,w,h) in eye_l:
        #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        bound_eye.append([x,y,w,h])
    for (x,y,w,h) in nose:
        #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        bound_nose.append([x,y,w,h])
    
    print(bound_eye)
    print(bound_nose)
    #bound_eye = []
    
    img1 = draw_glasses(img1,bound_eye)
    img1 = draw_mustache(img1,bound_nose)
    
    return img1



def show_img(img1):
    cv2.imwrite(data_path + 'Test\\After.png',img1) #Writing the modified image to the disk
    
    while True:
        cv2.imshow('img',img1)
        cv2.imshow('glasses',img1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
    print(img1.shape)
    nd = img1.reshape((img1.shape[0]*img1.shape[1] , -1))
    print(nd.shape)
    nd = nd.tolist()  #Since the nd is a numpy ndarray (is resized image which is made of pixels. Each pixel has BGR values (0-255 range))
    try:
        os.remove(data_path + 'Test\\output.csv') #if present remove the old csv file
    except():
        pass
    file = open(data_path + 'Test\\output.csv','w') #Empty csv file created.
    wr = csv.writer(file)                           
    wr.writerow(['b','g','r'])
    wr.writerows(nd)                               # We write the modified image's nd array values in our csv file.
    file.close()


#This is used for detecting eyes and nose and drawing glasses and moustache on them.
#Done by cv2's Video_Capture
def video_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose):
    cap = cv2.VideoCapture(0)
    cap.set(3, 720)  #Height (prop_id: 3) is set to 720p
    cap.set(4, 480)  #Width (prop_id: 4) is set to 480p
    
    while True:
        ret, img = cap.read()
        img = image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img) #This calls are functions and train each frame of our video
        cv2.imshow('img',img) 
     
        k = cv2.waitKey(1)
        if k  & 0xFF == ord('q'):
            break
    
    cap.release()  #Release the memory for the allotted video capture
    cv2.destroyAllWindows()




#video_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose)
img1 = cv2.imread(data_path + 'Test\\Before.jpg')
img1 = image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img1)
"""
img1 = cv2.imread('Data/Train/Jamie_Before.jpg')
img1 = image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img1)
img1 = cv2.resize(img1, (0,0), fx=0.5,fy=0.5)
cv2.imshow('My_image', img1)
cv2.waitKey(0)
cv2.DestroyAllWindows()
"""
show_img(img1)
