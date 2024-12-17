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
#cap = cv2.VideoCapture("Videos/cars.mp4")
cap =  cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 500)

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

button_person = False

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])

        is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
        if is_inside > 0:
            print("Clicking inside poly")
            print(x, y)

            if button_person is False:
                button_person = True
            else:
                button_person = False

            print("Now object detection is: ", button_person)


# Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)


while True:
    ret, frame = cap.read()

    # Object Detection
    (class_ids, score, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, score, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

    if class_name == 'person' and button_person is True:
        # Hand Detection
        hands, frame = detector.findHands(frame)

        cv2.putText(frame, class_name, (x, y -10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    # Create Button
    # cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 255), -1)
    polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
    cv2.fillPoly(frame, polygon, (0, 0, 200))
    cv2.putText(frame, "ON/OFF", (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
