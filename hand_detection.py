import cv2
import mediapipe as mp
#import time
import math
#import pyfirmata2


#board = pyfirmata2.Arduino("COM3")
#ledPin = board.get_pin("d:3:p")

cap = cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 500)

mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)

while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            handLandmarks = result.multi_hand_landmarks[0]
            thumbTip = handLandmarks.landmark[5]
            indexTip = handLandmarks.landmark[17]
            distance = math.sqrt((thumbTip.x - indexTip.x)**2 + (thumbTip.y - indexTip.y)**2)
            print(distance)

        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()