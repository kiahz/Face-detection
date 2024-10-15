import cv2
import tkinter as tk
from tkinter import messagebox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_eyes():
    cap = cv2.VideoCapture(0)
    detected = False
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) > 0:
                detected = True
                break

        cv2.imshow('Eye Detection', frame)

        if detected:
            messagebox.showinfo("Success", "Eyes detected! You are logged in.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_gui():
    root = tk.Tk()
    root.title("Eye Detection Login")
    root.geometry("300x150")

    label = tk.Label(root, text="Eye Detection Login System", font=("Arial", 14))
    label.pack(pady=20)

    start_button = tk.Button(root, text="Start Detection", command=detect_eyes, font=("Arial", 12))
    start_button.pack(pady=10)

    root.mainloop()

create_gui()