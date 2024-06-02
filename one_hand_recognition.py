import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import pickle

# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        ##    image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.resize(image, (500, 500))
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            n = len(results.multi_hand_landmarks)
            if n == 1:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                without_garbage = []
                clean = []
                for i in range(n):
                    data = results.multi_hand_landmarks[i]

                    data = str(data)

                    data = data.strip().split('\n')

                    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

                    for i in data:
                        if i not in garbage:
                            without_garbage.append(i)
                    for i in without_garbage:
                        i = i.strip()
                        clean.append(i[2:])

                    for i in range(0, len(clean)):
                        clean[i] = float(clean[i])

                Class = svm.predict(np.array(clean).reshape(-1, 63))
                Class = Class[0]
                print(Class)
                cv2.putText(image, str(Class), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 2)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
