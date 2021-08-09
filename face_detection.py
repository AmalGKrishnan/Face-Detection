import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# C:\\Users\\Reva\\Documents\\Python Scripts\\Face_Detection\\

webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()
    
    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
      
    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    #Draw rectangle around faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),1)
    
    cv2.imshow('Face Detector',frame)
    key = cv2.waitKey(1)
    
    #Stop if Q is pressed
    if key==81 or key==113:
        break

#Release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")

