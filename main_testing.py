from tkinter import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import statistics
import math
import gpt

import warnings
import os

import time


warnings.filterwarnings('ignore')

def main():
    root=Tk()
    root.title('MAIN PROGRAM')
    root['width']=400
    root['height']=200
    mainframe=Frame(root)
    mainframe.pack()
    blank=Frame(root,height=300,width=200)
    blank.pack()

    Label(mainframe,text=' ',font=('Times New Roman',30)).grid(column=0,row=0)
    Label(mainframe,text='  Eye & Head Movement Tracking System ',font=('Times New Roman',25)).grid(column=0,row=1)
    Label(mainframe,text=' ',font=('Times New Roman',30)).grid(column=0,row=2)
    Label(mainframe,text=' ',font=('Times New Roman',15)).grid(column=0,row=3)
    Label(mainframe,text=' ',font=('Times New Roman',30)).grid(column=0,row=4)


    def runmod1():
        root.destroy()
        import timing

        
        
 


    def runmod2():
        root.destroy()
        try:
            gpt.eye()
        except Exception:
            pass
       

    #def runmod3():
        #print('a')






    btmod1=Button(mainframe,text='Start recording',command=runmod1,width=25,height=3,font=('Times New Roman',15)).grid(row=6,column=0)
    Label(mainframe,text=' ',font=('Times New Roman',30)).grid(column=0,row=7)

    btmod2=Button(mainframe,text='Preview window',command=runmod2,width=25,height=3,font=('Times New Roman',15)).grid(row=8,column=0)
    Label(mainframe,text=' ',font=('Times New Roman',30)).grid(column=0,row=9)
    
    #btmod3=Button(mainframe,text='Motion Tracking',command=runmod3,width=25,height=3,font=('Times New Roman',15)).grid(row=10,column=0)


    root.mainloop()



main()
