import numpy as np
import cv2
import time
import os
import tkinter as tk
#
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


        
############
# prompt the user to enter a directory name
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






output_file = os.path.join(dir_name, 'output.mp4')

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# create a window
#root = tk.Tk()
#root.geometry("400x200")

# create a label and pack it into the window
#label = tk.Label(root, text="Enter a durations of motion capture (seconds):")
#label.pack()

# create a number input field and pack it into the window
#num_input = tk.Entry(root)
#num_input.pack()

# global variable to store the number value
#number = None


# function to get the value from the number input field
def get_number():
    global number
    number = float(num_input.get())
    print(f"The duration entered is {number}s")
    root.destroy()

# create a button to get the value from the number input field
#button = tk.Button(root, text="Set durations", command=get_number)
#button.pack()

# run the main event loop
#root.mainloop()

#capture_duration = number

print('Program Initialization...')

cap = cv2.VideoCapture(0)



width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))




fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

#start_time = time.time()




while(True):


   
    ret, frame = cap.read()
    
    
   

    if not ret:
        
        print("Imaging capture failed, please try again! ")
       
        break

    cv2.imshow('image_win',frame)
    #cap.set(cv2.CAP_PROP_FOCUS, )

    


    key = cv2.waitKey(1)
    
    if key == ord('q'):
        

      
        print("Preview has been terminated")
       
        break
   
   
    
    
 
    if key == ord('w'):
        subprocess.run('ATK_IMU1.04.exe', shell=True, check=True)
        
        break
    out.write(frame)




#while (cap.isOpened() and (time.time() - start_time) <= capture_duration):
  #  break

    #ret, frame = cap.read()
    
    

    #if ret == True:
      #  frame = cv2.flip(frame, 0)
        #out.write(frame)

    #else:
        #break

cap.release()

cv2.destroyAllWindows()
print('capture completed')




