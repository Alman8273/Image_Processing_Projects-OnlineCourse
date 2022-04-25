import cv2

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')

#cam opener
cap = cv2.VideoCapture(0)

while True:
   ret, img = cap.read()
   #rbg convert to gray
   g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   f = cascade_face.detectMultiScale(
       g,
       scaleFactor = 1.3,
       minNeighbors = 5,
       minSize = (30, 30)
   )
   
   

