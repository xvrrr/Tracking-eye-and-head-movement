import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import statistics
import math
import subprocess
import tkinter as tk
from PIL import Image, ImageTk
import main_testing






def eye():
    cap = cv2.VideoCapture(0)


    print("Check if the camera preview is launched? {}".format(cap.isOpened()))

    def show_frame():
        ret, frame = cap.read()
        #cap.set(cv2.CAP_PROP_BRIGHTNESS, 78)

        # Set the contrast to 30
        #cap.set(cv2.CAP_PROP_CONTRAST, 100)

        

        if not ret:
            print("Imaging capture stop!")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rows, cols, _ = frame.shape
        frame = frame[220:500, 100:400]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        _, threshold = cv2.threshold(gray_frame, 28, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(frame, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
            cv2.line(frame, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
            break

        frame = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=frame)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, show_frame)

    def close_window():
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        main_testing.main()

    root = tk.Tk()
    root.title("Object Detection Preview")
    root.geometry("800x600")

    video_frame = tk.Frame(root)
    video_frame.pack(side="left", padx=10, pady=10)

    video_label = tk.Label(video_frame)
    video_label.pack()

    button_frame = tk.Frame(root)
    button_frame.pack(side="left", padx=10, pady=10)

    exit_button = tk.Button(button_frame, text="Exit", command=close_window)
    exit_button.pack(side="bottom", pady=20)

    root.protocol("WM_DELETE_WINDOW", close_window)

    show_frame()
    root.mainloop()

    print(helpInfo)

    cap.release()

    cv2.destroyAllWindows()

eye()
