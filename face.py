import os
import cv2
import dlib
import sys
import numpy as np

detector = dlib.get_frontal_face_detector()


def read_img_to_rgb(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_rgb

smile=read_img_to_rgb(sys.argv[1])

#change the size of the smile image
smile = cv2.resize(smile, (500, 500))
cv2.imshow("Smile", smile)


#real-time face detection
cap = cv2.VideoCapture(0)
tempx=0
tempy=0
tempx1=0
tempy1=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        tempx=x
        tempy=y
        tempx1=x1
        tempy1=y1
       # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        cx= x + (x1-x)//2
        cy= y + (y1-y)//2
        radius = np.sqrt((x1-x)**2 + (y1-y)**2)/2
        cv2.circle(frame, (cx, cy), int(radius), (0, 255, 0), 2)
        #replace circle with smile image
        
        cur_face=smile
        cur_face=cv2.resize(cur_face, (int(radius*2), int(radius*2)))

        #get the region of interest
        

        
        
    
    if faces.__len__()==0:
         cv2.circle(frame, (tempx + (tempx1-tempx)//2, tempy + (tempy1-tempy)//2), int(np.sqrt((tempx1-tempx)**2 + (tempy1-tempy)**2)/2), (0, 255, 0), 2)
    
        
      

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

