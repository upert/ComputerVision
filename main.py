import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe
import math

cap = cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 500)

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=4)

#Loop
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']
        x1 = lmList[5]
        x2= lmList[17]


        print(x1, x2)




    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
