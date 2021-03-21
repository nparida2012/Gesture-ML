import os

#path='C:\\Users\\Paridnr\\Documents\\personal\\535_MC\\Moblie_Computing_Project_SourceCode_Nihar\\video_path\\'

def dir_crawler(path):
    list_of_files = []
    for (path,dir,files) in os.walk(path):
        for filename in files:
            #print(os.path.join(path,filename))
            list_of_files.append(os.path.join(path,filename))
    return(list_of_files)

