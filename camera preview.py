import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import statistics
import math













cap = cv2.VideoCapture(0)


print("Check if  the camera preview is launch? {}".format(cap.isOpened()))


#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


#cv2.namedWindow('image_win',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)


#cv2.namedWindow('image_win',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)


helpInfo = '''


Press Q： Program terminated


'''








while True:
    ret, frame = cap.read()


    
    if ret is False:
        break

    frame = frame[:, :]
    rows, cols, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    _, threshold = cv2.threshold(gray_frame, 28, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)



    

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    break




while True:
    ret, frame = cap.read()
    if ret is False:
        break

    frame = frame[:, :]
    rows, cols, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    _, threshold = cv2.threshold(gray_frame, 28, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    

    

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    

       
        

        

        
        


        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break








print(helpInfo)
while(True):


   
    ret, frame = cap.read()

    if not ret:
        
        print("Imaging capture failed, please try again! ")
       
        break

    cv2.imshow('image_win',frame)

    


    key = cv2.waitKey(1)
    
    if key == ord('q'):
      
        print("Preview has been terminated")
        
        break
    
 
    if key == ord('w'):
        subprocess.run('ATK_IMU1.04.exe', shell=True, check=True)
        
        break










    
cap.release()

cv2.destroyAllWindows()
