import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import statistics
import math



datasetx = []
datasety = []

p1=[]
p2=[]

lis= []

offset1 =0
offset2 =0

cap = cv2.VideoCapture("28.mp4")
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

 
#plot_cicle()











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

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        for i in range(1):
            offset1 = (x+x+w)/2
            offset2 = ((y+y+h)/2)+680
            break

        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 4)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 4)
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

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
    
        axis1 = (x+x+w)/2
        axis2 = ((y+y+h)/2)+680
        p1.append((axis1))
        p2.append((axis2))

        axis1sd = axis1-offset1
        axis2sd = axis2-offset2
        datasetx.append(axis1sd)
        datasety.append(axis2sd)


       
        

        

        
        
        print(axis1,axis2)
       
        plt.scatter(axis1,axis2,marker='o', color="red", s=50)
        
        
        


        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 4)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 4)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break









#####

variancex=statistics.pvariance(datasetx)
variancey=statistics.pvariance(datasety)

print(variancex)
print(variancey)
sample = math.sqrt((variancex+variancey)/2)


total = 0
 
for ele in range(0, len(p1)):
    total = total + p1[ele]

RMS = math.sqrt((total*0.3)/len(datasetx))
print("RMS is")
print(RMS)
print("above")



print("Standard Deviation of sample is % s "% (sample))



###


plt.plot(p1,p2)
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.savefig("scanning.jpg")




###


plt.figure()
newx = range(len(p1))

# 绘制折线图

plt.scatter(newx, p1,marker='o', color="red", s=5)
plt.plot(newx, p1)
plt.xlabel('frames')
plt.ylabel('x')
plt.title('x wave Plot')
plt.savefig("x wave.jpg")
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
plt.savefig("y wave.jpg")
# 显示图形
plt.show()




cap.release()
cv2.destroyAllWindows()




