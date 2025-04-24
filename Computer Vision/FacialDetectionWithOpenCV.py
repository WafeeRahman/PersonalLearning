#Facial Recognition Demo 

# Imports
import cv2 


#Load in Cascades, these cascades will detect haar-like features. There are no neural nets required, since the XMLs contain the features we need, and are built into openCV
#The cascade F(x) is a series of filters and weak features that we apply for detection (stored in xml)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Import facial feature cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #Import facial feature cascade



""" 
Defining a function that detect faces through logitech camera.

This function will draw squares around faces, aswell as detect eye features and faces

Input - Face Image (Frames of Video Coming From Webcam) in grayscale. aswell as the frame of the image

Output - Outline Image Ontop of Face 

"""
def detect(gray, frame):
    #Import detectMultiScale function
    #Pass in grayscale image, decrease it by 1.3x, and have acceptance criteria of 5 features in adaboost
    #Experimental features - we find that these parameters work best for face detection through experimentation
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #For the tuple sets given by faces
    for x,y,w,h in faces:
        #Draw rectangle over our face frame
        #Coordinates of our face detected by multiscale filters
        #Face Frame, Face Upper Left Coordinates, Bottom Right Coordinate, Frame Colour, thickness

        cv2.rectangle(frame,(x,y), (x+w, y+h), (255, 0, 0), 2) #Prints Rectangle Into Frame
        
        #Two regions of interest for eye features, one for the grayscale image passed in, and another for the original
        #Region of Interest is the range between x, y and bottom right
        roi_gray = gray[y:y+h,  x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #Follow the same steps for eye cascade using the detectMultiScale provided by eye_cascade xml
        #We pass in our region of interest instead, to save computation time and detect eyes exclusively within region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2) #ROI Color Prints into Frame Aswell, as it works with the subsequence of frame
    return frame

# Connect to Camera and Feed Frames into Function
video_capture = cv2.VideoCapture(0) #Pass in EXT Logitech Webcam
while True:
    #Get Last Frame
    _, frame = video_capture.read()
    #Tranform to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert RGB to Gray
    canvas = detect(gray, frame) 
    cv2.imshow('Video', canvas) #Display to user
    
    #If we keyboard input q, quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()