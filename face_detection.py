import cv2 as cv

path = r'./assets/picture/samplePicture.jpg'

img = cv.imread(path)

#print (img)
# make picture gray
grayPicture = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#print(grayPicture)

#cv.imshow('picutre',grayPicture)

# set classifier 
haar_face_classifier = cv.CascadeClassifier('haar_face.xml')

# use classifier to detect faces 
faces_rect = haar_face_classifier.detectMultiScale(grayPicture , scaleFactor=1.04 , minNeighbors=3)


print("Coordinates of faces:\n")
print(faces_rect)
print("Wait for picture to popup...")

for (x,y,w,h) in faces_rect:
    cv.rectangle(img , (x,y) , (x+h,y+w) , (255,0,0) , thickness=1)

cv.imshow('Detected Person',img)

cv.waitKey(0)