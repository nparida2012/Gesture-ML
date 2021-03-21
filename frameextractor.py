# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
import crawler

#videopath : path of the video file
#video_path='C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\video_path\\'
#frames_path: path of the directory to which the frames are saved
#frames_path='C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\frames_path\\'
#count=1

def frameExtractor(video_path,frames_path,count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    list_of_files=crawler.dir_crawler(video_path)
    for filepath in list_of_files:
        videopath=filepath 
        #print(videopath)
        cap = cv2.VideoCapture(videopath)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        frame_no= int(video_length/2)
        #print("Extracting frame..\n")
        cap.set(1,frame_no)
        ret,frame=cap.read()
        filename= os.path.basename(videopath)
        #print(filename)
        name = filename.split(".")[0]
        #print(name)
        rotated_frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #ROTATE_90_COUNTERCLOCKWISE
        frames_file=str(os.path.join(frames_path,name+".png"))
        cv2.imwrite(frames_file, rotated_frame)
        count=count+1
    return
