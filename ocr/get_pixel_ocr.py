# import the necessary packages
import imutils
import cv2
import joblib
import numpy as np

import math
import utm
import matplotlib.pyplot as plt
import os

MODEL_NAME = 'shop'
ROOT_SAVE_FOLDER = "/home/itis/Desktop/Work_Flow_OCR/src/store/data/Allowed_Classes/prediction3/"


pixel_path_dir = os.path.join(ROOT_SAVE_FOLDER)
f = open ('Pixel_info_OCR_{}.csv'.format(MODEL_NAME), 'w')

# ROOT_SAVE_FOLDER = "/home/itis/jaipur_new_las/PANO/test_skel/"

#MISSION_DIR = "2021-FEB-25_Mission2CGSML"

# BIN_FOLDER = "/home/itis/jaipur_new_las/PANO/test_skel/eval_test/"


# bin_dir = os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME,MISSION_DIR, TRACK_FOLDER_FOLDER, BIN_FOLDER) # 

# for bin_dir in os.listdir(os.path.join(ROOT_SAVE_FOLDER)):

if True:#for MISSION_FOLDER in os.listdir(os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME)):
    MISSION_FOLDER = 'Mission_OCR'
    # if 'Mission' not in MISSION_FOLDER:
    #     continue
    if True:#for TRACK_FOLDER in os.listdir(os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME, MISSION_FOLDER)):

        for BIN_FOLDER in os.listdir(os.path.join(ROOT_SAVE_FOLDER)):
            if 'eval' not in BIN_FOLDER:
                continue

            bin_dir = os.path.join(ROOT_SAVE_FOLDER,BIN_FOLDER) #os.path.join(ROOT_SAVE_FOLDER, bin_dir)
            for bin_file in os.listdir(bin_dir):
                if '.bin' in bin_file:
                    print (bin_file)
                    TRACK_FOLDER = bin_file.split('-')[0]
                    x = joblib.load(os.path.join(bin_dir, bin_file))
                    valid = False
                    try:
                        IMAGE_NAME = x[0]['imageName']
                        print (IMAGE_NAME)
                        thing_classes = x[1]
                        imageWidth, imageHeight =x[0]['imgShape'][1], x[0]['imgShape'][0] #img.shape[1], img.shape[0]
                        point_str = ""
                        # for num, box in enumerate(x[0]['predBoxes']):
                        #     # print ("Class", x[0][0]['predClasses'])
                        #     # print(x[0]['boxScoresdraw_instance_predictions'])
                        #     # print ("Class Name", thing_classes[x[0]['predClasses'][num]])                            
                        #     targetPixelX =  int((box[0]+box[2])/2)
                        #     targetPixelY = int(box[3])
                        #     if not inside_aoi(targetPixelX, targetPixelY, imageHeight, imageWidth):
                        #         print ("Not Inside AOI")
                        #         continue

                        #     aoi_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]

                        #     output = reader.readtext(aoi_img, detail = 0, paragraph=True, contrast_ths =0.3)
                        points_list = x[0]['instance_predictions']
                        print (points_list)

                        for num, points in enumerate(points_list):
                            point_str = ""
                            box_str = ""
                            ASSET, box ,points = points
                            box_str = str(box[0])+","+str(box[1])+","+str(box[2])+","+str(box[3])

                            for point in points:
                                point_str = point_str + str(point[0]) +','+ str(point[1])+';'
                            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(bin_dir, BIN_FOLDER, MISSION_FOLDER,IMAGE_NAME, imageWidth, imageHeight, ASSET, box_str, point_str))

                        # for num, box in enumerate(x[0]['predBoxes']):
                        #     valid = True
                        #     ASSET = x[0]['predClasses'] [num]
                        #     object_mask = x[0]['instance_predictions'][num]
                        #     object_mask = np.array(object_mask)
                        #     print (object_mask, IMAGE_NAME, ASSET)
                        #     point_str = ""
                        #     for point in object_mask:
                        #         point_str = point_str + str(point[2]) +','+ str(point[1])+';'

                            # f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(ROOT_SAVE_FOLDER, BIN_FOLDER, IMAGE_NAME, imageWidth, imageHeight, ASSET, point_str))
                    except Exception as E:
                        print (E)
                        pass
                    # if valid:
                    # 	break
                    


f.close()


