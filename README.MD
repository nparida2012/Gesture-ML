As a highlevel approach greyscale images are captured from the train videos and converted 3 basis numpy vectors. These vectors are trained using CNN model and appropriately labelled. The test videos without the labels are then used to derive greysale image and corresponding numpy vectors. These test vector are passed through the CNN Model and their cosine similarities then observed with each of the trained vectors. The lowest match is then determined as the classified label.


**File Structure:**
Make sure the traindata folder has the below subfolders
•	Training and Testing videos folders are
o	Train_video_path: This will store all the gesture videos that will be used to train the cnn model 
o	Test_video_path : This will store all the test gesture videos that will be used for classification using the trained model
•	Training and Testing image folders are
o	Train_Frames_path: grey scale Train images will be derived using the opencv library from the train videos
o	Test_frames_path: grey scale test images will be derived using the opencv library from the test videos

**•	Image file name to classfification reference file is used**
o	Gesture_map_input.csv: This is a reference file where we will store the train video name and corresponding classification label. 

Code Segments:
•	Crawler.py : It is used to crawl the local directories for training and testing videos and images
•	Frameextractor.py:It is used to extract penultimate image from the gesture video and save it to the corresponding frames path
•	Handshape_feature_extractor.py: It is used to preprocess the images and call the cnn model for classification of both train and test videos

**Libraries used:**
•	tensorflow              2.4.1
•	Keras-Preprocessing     1.1.2
•	pandas                 	1.1.0
•	numpy                   1.20.1
•	opencv-python          	4.4.0.44
