import os
import cv2
import cvzone
import pickle
import numpy as np
import face_recognition
from face_recognition import face_encodings
from EncoderDetector import encodingslistwithstids

imagespath = 'Images'
imgpathlist = os.listdir(imagespath)


data = {
    "321654":
        {
            "name": "Murtaza Hassan",
            "major": "Robotics",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34",
            "img": cv2.imread(os.path.join(imagespath,imgpathlist[1]))
        },
    "852741":
        {
            "name": "Emly Blunt",
            "major": "Economics",
            "starting_year": 2021,
            "total_attendance": 12,
            "standing": "B",
            "year": 1,
            "last_attendance_time": "2022-12-11 00:54:34",
            "img": cv2.imread(os.path.join(imagespath,imgpathlist[2]))
        },
    "963852":
        {
            "name": "Elon Musk",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34",
            "img": cv2.imread(os.path.join(imagespath,imgpathlist[3]))
        },
"100000":
        {
            "name": "Prashant Pratap Singh",
            "major": "Computers",
            "starting_year": 2021,
            "total_attendance": 8,
            "standing": "G",
            "year": 1,
            "last_attendance_time": "2022-12-11 00:54:34",
            "img": cv2.imread(os.path.join(imagespath,imgpathlist[0]))
        }
}
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList =[]
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))



file =open("EncodeFile.p","rb")
encodingslistwithstids=pickle.load(file)
file.close()

encodingslist,stids = encodingslistwithstids

counter = 0
imgcounter = []
while True:
    success,img = cap.read()
    imgbackground = cv2.imread('Resources/background.png')

    imgbackground[162:162 + 480, 55:55 + 640] = img
    img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    currframeloc = face_recognition.face_locations(img)
    currfaceenc = face_recognition.face_encodings(img, currframeloc)



    for frameloc,faceenc in zip(currframeloc,currfaceenc):
        matches = face_recognition.compare_faces(encodingslist,faceenc)
        facedist = face_recognition.face_distance(encodingslist,faceenc)

        minindex = np.argmin(facedist)
        id = stids[minindex]


        if matches[minindex] and counter<=10:
            y1,x2,y2,x1 = frameloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            bbox = x1+55,y1+162,x2-x1,y2-y1
            cvzone.cornerRect(imgbackground,bbox,rt=0)
            imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[1]
            cv2.putText(imgbackground,f'{id}',(1008,493),cv2.FONT_HERSHEY_COMPLEX,
                    0.5,(255,255,255),1)
            cv2.putText(imgbackground, f'{data[str(id)]['major']}', (1008, 551),
                        cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 255), 1)
            cv2.putText(imgbackground, f'{data[str(id)]["standing"]}', (915, 624), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (100,100,100), 1)
            cv2.putText(imgbackground, f'{data[str(id)]["year"]}', (1025, 624), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (100,100,100), 1)
            cv2.putText(imgbackground, f'{data[str(id)]["starting_year"]}', (1123, 624), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (100,100,100), 1)
            cv2.putText(imgbackground, f'{data[str(id)]["total_attendance"]}', (867, 119), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (0, 255, 255), 1)
            w = cv2.getTextSize(f'{data[str(id)]["name"]}',cv2.FONT_HERSHEY_COMPLEX,0.7,1)[0][0]
            offset = (414-w)//2
            cv2.putText(imgbackground, f'{data[str(id)]["name"]}', (808+offset, 442), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (255, 0, 255), 1)
            imgbackground[176:176 + 216, 908:908 + 216] = data[str(id)]["img"]
            counter+=1
            imgcounter.append(id)

        elif 20>=counter>10:
            imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[2]
            counter+=1

        elif 30>=counter>20:
            imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[0]
            counter+=1
        else:
            counter=0



    cv2.imshow("Attendance",imgbackground)
    cv2.waitKey(1)

