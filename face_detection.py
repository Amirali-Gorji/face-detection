import cv2 as cv

path = r'D:\programming\Face detection\src\pictures\train\hajghasem\19.jpg'

img = cv.imread(path)

#print (img)
grayPicture = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#print(grayPicture)

#cv.imshow('picutre',grayPicture)

haar_face_classifier = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_face_classifier.detectMultiScale(grayPicture , scaleFactor=1.04 , minNeighbors=3)

#print(f'number is {len(faces_rect)}')

print(faces_rect)


for (x,y,w,h) in faces_rect:
    cv.rectangle(img , (x,y) , (x+h,y+w) , (255,0,0) , thickness=1)

cv.imshow('Detected Person',img)

cv.waitKey(0)