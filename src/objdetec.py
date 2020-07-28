#import necessary libraries
import cv2
import struct
import numpy as np
import time
import serial
#set the number of minimum features to be matched
MIN_MATCH_COUNT=20
#open SIFT feature
detector=cv2.xfeatures2d.SIFT_create()
#indexing and opening of Flann based functions
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
#read the reference image
trainImg=cv2.imread("TrainingData/TrainImg6.jpg",0)
#store the keypoints and their description
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
#initialize required distance and width of reference image
knownWidth = 9.6
knownDistance = 24
flag = 0
#calculate the width in pixels
pixelWidth = trainImg.shape[1] #426
#print(pixelWidth)
#calculate focal lenght of camera
focalLength = (pixelWidth*knownDistance)/knownWidth
#connect to arduino via serial communication
arduino = serial.Serial('COM6',9600,timeout=0.1)
time.sleep(2)
#open camera for real time video
cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read() #read frames from camera
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)   #convert frames to grayscale 
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)  #store keypoints and description of grayscale frame
    matches=flann.knnMatch(queryDesc,trainDesc,k=2) #match descriptions of frame with reference image

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)     #if match > 75% append the match and calculate for minimum number of matches
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)   #append keypoints of reference image after match
            qp.append(queryKP[m.queryIdx].pt)   #append keypoints of frame after match
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        #print(h,w)
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]]) #find coordinates of the box
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        #print(queryBorder)
        #print(queryBorder,"--\n--",queryBorder[0],"--\n--",queryBorder[0][0],"--\n--",queryBorder[0][0][1])
        #find width of frame in pixels
        width = abs(((queryBorder[0][3][0]- queryBorder[0][0][0])+(queryBorder[0][2][0] - queryBorder[0][1][0]))/2)
        #find distance of object from camera lens
        distance = int((focalLength*knownWidth)/width)
        if distance > 20 and flag == 0:
            flag = 1
            arduino.write(struct.pack('>B',flag))
            #print("Motor will move for ",distance)

        elif distance < 20 and flag == 1:
            flag = 0
            arduino.write(struct.pack('>B',flag))
            #print("Motor Stopped")
        

        #create the boundary box on screen and put distance
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        cv2.putText(QueryImgBGR,"%d in"%(distance),(QueryImgBGR.shape[1]-200, QueryImgBGR.shape[0]-20 ),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,0,0),3)
    else:
        print ("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        arduino.write(struct.pack('>B',0))
        break
#close all functions and camera
cam.release()
arduino.close()
cv2.destroyAllWindows()
