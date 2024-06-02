import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from PIL import Image, ImageTk

# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class HandRecognitionApp:
    def __init__(self, root, cap):
        self.root = root
        self.cap = cap

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start_processing)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_processing)
        self.stop_button.pack()

        self.processing = False
        self.process_video()

    def start_processing(self):
        self.processing = True

    def stop_processing(self):
        self.processing = False

    def process_video(self):
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while True:
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.resize(image, (500, 500))
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if self.processing and results.multi_hand_landmarks:
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

                        class_result = svm.predict(np.array(clean).reshape(-1, 63))
                        class_result = class_result[0]
                        print(class_result)
                        cv2.putText(image, str(class_result), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 2)

                # Display the processed image in the Tkinter window
                self.display_image(image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def display_image(self, image):
        # Convert the OpenCV image to a Tkinter PhotoImage
        img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new image
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.root.update_idletasks()

# Create a Tkinter window
root = tk.Tk()
root.title("Hand Recognition App")

# For webcam input:
cap = cv2.VideoCapture(0)

# Create an instance of the HandRecognitionApp class
app = HandRecognitionApp(root, cap)

# Run the Tkinter event loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()





