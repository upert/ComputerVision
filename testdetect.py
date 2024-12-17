import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load Class list
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# Initialize camera
cap =  cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 500)

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Create window
cv2.namedWindow("Frame")

while True:
    ret, frame = cap.read()

    # Object Detection
    (class_ids, score, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, score, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        hands, frame = detector.findHands(frame)

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break