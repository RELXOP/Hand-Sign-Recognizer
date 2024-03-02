#Importing all Libraries
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)               #Camera Acces
detector = HandDetector(maxHands=1)     #How Many Hands it can track
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")     #Loading Models
offset = 20    #Just For Adjustment
imgSize = 300  #Camera Size
counter = 0    #Counter for Image
pTime = 0      #Past Time for Fps Calculation
cTime = 0      #Curect Time for Fps Calculation
labels = ["A", "B", "C", "I Love U", "Mini Heart", "Ok", "Thumbs Up", "Otat Mea"]   #Signs Shown In Screen
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)   #Hand dectector Working on the Camera

####### FPS METER ########
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
##########################

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)   #It Predicts the outcome of the sign we make by our hand
            print(prediction, index)                                             #It Prints Prediction in the command line
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 192, 203), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 192, 203), 4)
       # cv2.imshow("ImageCrop", imgCrop)              #It Show the Box of my recognized Hand
       # cv2.imshow("ImageWhite", imgWhite)            #It Show Same as Above but in a presentable way
    cv2.imshow("Minor Project", imgOutput)    #It is the output we need
    cv2.waitKey(1)                                    #Delay For Camera to work Properly