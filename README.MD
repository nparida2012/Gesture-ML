File Structure:
•	Training and Testing videos folders are
o	Train_video_path
o	Test_video_path
•	Training and Testing image folders are
o	Train_Frames_path
o	Test_frames_path
•	Image file name to classfification reference file is used
o	Gesture_map_input.csv
Code Segments:
•	Crawler.py : It is used to crawl the directories for training and testing videos and images
•	Frameextractor.py:It is used to extract penultimate image from the gesture video and save it to the corresponding frames path
•	Handshape_feature_extractor.py: It is used to 
Libraries used:
•	tensorflow              2.4.1
•	Keras-Preprocessing     1.1.2
•	pandas                 	1.1.0
•	numpy                   1.20.1
•	opencv-python          	4.4.0.44
