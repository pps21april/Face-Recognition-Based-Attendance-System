import os
import cv2
import cvzone
import face_recognition
import pickle


imagespath = 'Images'
imgpathlist = os.listdir(imagespath)
imglist=[]
stids =[]
for path in imgpathlist:
    imglist.append(cv2.imread(os.path.join(imagespath,path)))
    stids.append(os.path.splitext(path)[0])

encodingslist=[]

for img in imglist:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(img)[0]
    encodingslist.append(encoding)

encodingslistwithstids = [encodingslist,stids]

file = open("EncodeFile.p","wb")
pickle.dump(encodingslistwithstids,file)
file.close()


