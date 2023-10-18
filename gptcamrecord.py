import cv2

import time
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import statistics
import math
import subprocess
from PIL import Image, ImageTk
import time



def get_dir_name():
    dir_name = input("Enter a user folder name: ")
    return dir_name

# get the directory name from the user
dir_name = get_dir_name()

try:
    # create the directory
    os.mkdir(dir_name)
except FileExistsError:
    # if the directory already exists, ask the user if they want to overwrite it
    answer = input(f"The directory '{dir_name}' already exists. Do you want to overwrite it? (y/n) ")
    if answer.lower() == 'y':
        # if the user answers 'y', delete the existing directory and create a new one
        os.rmdir(dir_name)
        os.mkdir(dir_name)
    else:
        # if the user answers 'n', exit the program
        exit()

# function to access the directory
def access_directory():
    # list the contents of the directory
    contents = os.listdir(dir_name)
    print(f"Contents of folder '{dir_name}':")
    for item in contents:
        print(item)

# call the function to access the directory
access_directory()


helpInfo = '''


Press Q： Program terminated




'''




output_file = os.path.join(dir_name, 'output.mp4')
absoutput_file = os.path.abspath(dir_name)


if not os.path.exists(dir_name):
    os.makedirs(dir_name)


subprocess.Popen(r'''C:\Users\13207\Desktop\fypsoftware\testing\ATK_IMU1.04.exe''')
time.sleep(5)
print('start')
cap = cv2.VideoCapture(0)

print("Check if  the camera preview is launch? {}".format(cap.isOpened()))
print('Loading......')
print('helpinfo')


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)



width = int(cap.get(3)) 
height = int(cap.get(4))



# 创建一个VideoWriter对象，用于保存视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编码方式
out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))  # 设置输出文件名和帧率

# 开始录制
while True:
    ret, frame = cap.read()  # 捕捉一帧视频
   
    if not ret:
        break
    
    cv2.imshow('frame', frame)  # 显示当前帧
    

    # 检测键盘输入，如果按下q键则停止录制并保存视频
    if cv2.waitKey(1) == ord('q'):
        break

    # 将当前帧写入输出视频文件
    
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print('Capture completed')


####################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import statistics
import math


import time


datasetx = []
datasety = []

if len(datasetx) > 0:
    variancex = statistics.pvariance(datasetx)
else:
    print("Datasetx is ready")

if len(datasety) > 0:
    variancex = statistics.pvariance(datasety)
else:
    print("Datasety is ready")

p1=[]
p2=[]

lis= []

offset1 =0
offset2 =0

#cap = cv2.VideoCapture(output_file,"output.mp4")


# 打开视频文件
cap = cv2.VideoCapture(absoutput_file + "/output.mp4")

#cap = cv2.VideoCapture(0)












def plot_cicle():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cir0 = Circle(xy = (540, 960), radius=10, alpha=0.2, color='b')
    cir1 = Circle(xy = (540, 960), radius=25, alpha=0.4, color='g')
    cir2 = Circle(xy = (540, 960), radius=40, alpha=0.3, color='m')
    cir3 = Circle(xy = (540, 960), radius=55, alpha=0.2, color='y')
    ax.add_patch(cir1)
    ax.add_patch(cir2)
    ax.add_patch(cir3)
    ax.add_patch(cir0)
    x, y = 540, 960
    ax.plot(x, y, '+')
    x, y = 1080,0
    ax.plot(x, y, '+')
    x,y = 0,1920
    ax.plot(x, y, '+')
    x,y = 0,0
    ax.plot(x, y, '+')
    x,y = 1080,1920
    ax.plot(x, y, '+')



    plt.axis('scaled')
    plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length

 
plot_cicle()











while True:
    ret, frame = cap.read()


    
    if ret is False:
        break

    frame = frame[680:1500, :]

    rows, cols, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    _, threshold = cv2.threshold(gray_frame, 28, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)



    

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for i in range(1):
            offset1 = (x+x+w)/2
            offset2 = ((y+y+h)/2)+680
            break

        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    break




while True:
    ret, frame = cap.read()
    if ret is False:
        break

    frame = frame[680:1500, :]

    rows, cols, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    _, threshold = cv2.threshold(gray_frame, 28, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    

    

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        axis1 = (x+x+w)/2
        axis2 = ((y+y+h)/2)+680
        p1.append((axis1))
        p2.append((axis2))

        axis1sd = axis1-offset1
        axis2sd = axis2-offset2
        datasetx.append(axis1sd)
        datasety.append(axis2sd)

       
        

        

        
        
        print(axis1,axis2)
        plt.scatter(axis1,axis2,marker='o', color="red", s=40)
        
        
        


        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break




variancex=statistics.pvariance(datasetx)
variancey=statistics.pvariance(datasety)
print(variancex)
print(variancey)
sample = math.sqrt((variancex+variancey)/2)
mmsample = sample*0.3
print("Standard Deviation of sample is % s "% (sample))

#########


with open(os.path.join(absoutput_file + "/output.txt"), 'w') as f:
    f.write(f"variancex: {variancex}\n")
    f.write(f"variancey: {variancey}\n")
    f.write(f"SD(pixel): {sample}\n")
    f.write(f"SD(mm): {sample}\n")



##########

###

plt.plot(p1,p2)
plt.xlabel('pixels')
plt.ylabel('pixels')



plt.savefig(absoutput_file + "/scanning.jpg")


###


plt.figure()
newx = range(len(p1))

# 绘制折线图

plt.scatter(newx, p1,marker='o', color="red", s=5)
plt.plot(newx, p1)
plt.xlabel('frames')
plt.ylabel('x')
plt.title('x wave Plot')

plt.savefig(absoutput_file + "/x wave.jpg")
# 显示图形



####

plt.figure()
newy = range(len(p2))

# 绘制折线图

plt.scatter(newy, p2,marker='o', color="red", s=5)
plt.plot(newy, p2)
plt.xlabel('frames')
plt.ylabel('y')
plt.title('y wave Plot')

plt.savefig(absoutput_file + "/y wave.jpg")
# 显示图形
plt.show()





plt.show()
cap.release()
cv2.destroyAllWindows()














