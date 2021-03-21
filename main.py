# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import frameextractor as fe
import handshape_feature_extractor
import crawler
import pandas as pd

#videopath : path of the video file

#base_dir= os.getcwd()
base_dir = os.path.dirname(os.path.abspath(__file__))
traindata='traindata'
testdata='test'
train_frames_path='train_frames_path'
test_frames_path='test_frames_path'

training_video_path= os.path.join(base_dir,traindata)
#'C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\training_video_path\\'
#frames_path: path of the directory to which the frames are saved
training_frames_path=os.path.join(training_video_path,train_frames_path)      
#'C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\training_frames_path\\'

#videopath : path of the video file
test_video_path= os.path.join(base_dir,testdata) 
#'C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\test_video_path\\'
#frames_path: path of the directory to which the frames are saved
test_frames_path= os.path.join(training_video_path,test_frames_path) 
#'C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\test_frames_path\\'

mapping_file = os.path.join(base_dir,'gesture_map_input.txt')
#'C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\gesture_map_input.csv'

output_file = os.path.join(base_dir,'Results.csv')
#'C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\results.csv'

df=pd.read_csv(mapping_file)

#print(training_video_path)
#print(training_frames_path)
#print(test_video_path)
#print(test_frames_path)

print(df)

#sys.exit(0)

## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
print('generate training images...')
fe.frameExtractor(training_video_path,training_frames_path,1)
list_of_files=crawler.dir_crawler(training_frames_path)
#print(list_of_files)
X_train=[]
y_train=[]
train_file_names=[]
print('for each training image, assign label...')
for image_file in list_of_files:
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    #print('Original Dimensions : ',img.shape)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print('gray scale Dimensions : ',img_gray.shape)
    feature = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance()
    feature_vector=feature.extract_feature(img_gray)
    X_train.append(feature_vector)
    file_name=os.path.basename(image_file)
    #print(file_name)
    name=file_name.split(".")[0]
    #print(name)
    train_file_names.append(name)
    y_label=df[df['train_gesture_name']==name]['output_label'].values[0]
    y_train.append(y_label)

train_df=pd.DataFrame({'file_name':train_file_names,'X_train':X_train,'y_train':y_train})
print('train vector...')
print(train_df)

#sys.exit(0)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
print('generate testing images...')
test_feature_vector=[]
fe.frameExtractor(test_video_path,test_frames_path,1)
list_of_files=crawler.dir_crawler(test_frames_path)
X_test=[]
test_file_names=[]
gesture_name=[]
print('for each testing image, build test vector...')
for image_file in list_of_files:
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    #print('Original Dimensions : ',img.shape)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print('gray scale Dimensions : ',img_gray.shape)
    feature = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance()
    feature_vector=feature.extract_feature(img_gray)
    X_test.append(feature_vector)
    file_name=os.path.basename(image_file)
    file_name = file_name.split(".")[0]
    test_file_names.append(file_name)
    #gesture=df[df['test_gesture_name']==file_name]['gesture'].values[0]
    #gesture_name.append(gesture)
    #y_label=df[df['test_gesture_name']==name]['output_label'].values[0]
    #y_test.append(y_label)
test_df=pd.DataFrame({'file_name':test_file_names,'X_Test':X_test})
print('test vector..')
print(test_df)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
y_test=[]
y_loss=[]
print('for each test vector item derive, loss factor and label with closest similarity ..')
for test_vector in X_test: # 34 test image
    test_train_loss=[]
    test_train_label=[]
    #print(item)
    i=0
    for file_name in train_df['file_name'].to_list(): # 17 train items, each test item is compared with trained items
        train_vector = train_df['X_train'].values[i]
        train_label = train_df['y_train'].values[i]
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        test_train_loss.append(cosine_loss(test_vector, train_vector).numpy()) # list of 17 losses corresponding to each test image
        #val=train_df[train_df['X_train']==vector]['y_train'].values[0]
        test_train_label.append(train_label) # corresponding train image
        i=i+1
    res = dict(zip(test_train_label,test_train_loss)) 
    y_loss.append(min(res.values())) # minimum cosine loss
    y_test.append(min(res,key=res.get)) # find the label with minimum loss
    #print('min loss={}, min label={}'.format(min(res.values()),min(res,key=res.get)))
test_df['loss']=y_loss
test_df['Output Label']=y_test
result_df=test_df[['Output Label']]
print('result file contents..')
print(test_df)
result_df.to_csv(output_file, header=None,index=False)  
