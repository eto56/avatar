import cv2
from deepface import DeepFace
import sys

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

smile = cv2.imread(sys.argv[1])
#cv2.imshow("Smile", smile)
#change the shape of the smile image to a circle

surprise=cv2.imread(sys.argv[2])
neutral=cv2.imread(sys.argv[3])


# Start capturing video
cap = cv2.VideoCapture(0)
cnt=0
cur_emotion="neutral"
sum=0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    for (x, y, w, h) in faces:
        cnt+=1
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
       
        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        sub=result[0]['emotion']

        happy_score=result[0]['emotion']['happy']
        surprise_score=result[0]['emotion']['surprise']
        neutral_score=result[0]['emotion']['neutral']
        fear_score=result[0]['emotion']['fear']
        sad_score=result[0]['emotion']['sad']
        angry_score=result[0]['emotion']['angry']
        disgust_score=result[0]['emotion']['disgust']
        # print(sub)
        if cnt%10000==0:
            cur_emotion=emotion
    
        cv2.circle(frame, (x + w//2, y + h//2), int((w+h)/4), (0, 255, 0), 2)

        # Draw rectangle around face and label with predicted emotion
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #replace face with smile image
        if cur_emotion=="happy" or happy_score>2:
            cur_face=smile
        elif cur_emotion=="surprise" or surprise_score>2:
            cur_face=surprise
        else:
            cur_face=neutral
        sum+=happy_score
        cur_face=cv2.resize(cur_face, (w, h))
        frame[y:y + h, x:x + w]=cur_face

        cv2.putText(frame, cur_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        score=sum/cnt
        cv2.putText(frame, "Happy: "+str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)
 

    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Press 'esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release the capture and close all windows

cap.release()


cv2.destroyAllWindows()

