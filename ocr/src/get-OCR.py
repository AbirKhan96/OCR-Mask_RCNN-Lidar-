# import the necessary packages
import imutils
import cv2
import joblib
import numpy as np

import math
import utm
import matplotlib.pyplot as plt
import os

import easyocr
import os
reader = easyocr.Reader(["en",'hi'], gpu = True)



# ROOT_SAVE_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/AI_output/"
# MODEL_NAME = "electric_pole"
# MISSION_DIR = "2021-FEB-25_Mission2CGSML"

BIN_FOLDER = "/home/itis/Desktop/Work_Flow_OCR/src/store/data/Allowed_Classes/prediction/eval_Images/"
BIN_FOLDER = "/home/itis/Desktop/Work_Flow_OCR/src/store/data/Allowed_Classes/prediction/eval_All_SHOP_ANNOTATION/"
IMAGE_FOLDER = "/home/itis/Desktop/Work_Flow_OCR/Test image for ocr/ToAbirFromHemant/Images/"   #"/home/itis/Desktop/AI_jaipur_Output/ocr_images/"/home/itis/Desktop/Work_Flow_OCR/Test image for ocr/ToAbirFromHemant/Images/
IMAGE_FOLDER = "/home/itis/Desktop/Work_Flow_OCR/All_SHOP_ANNOTATION/"
# pixel_path_dir = os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME, MISSION_DIR)




out_csv = open("SHOP_JAIPUR_OCR_OUTPUT_NEW.csv", 'a', encoding = 'utf-8')
# f = open (pixel_path_dir + 'pixel_info.csv', 'w')
# bin_dir = os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME,MISSION_DIR, TRACK_FOLDER_FOLDER, BIN_FOLDER) # 
if True:
	if True:
		bin_dir = BIN_FOLDER #os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME, MISSION_DIR, TRACK_FOLDER, bin_dir)
		# bin_dir = '../bin_data/te2/eval_TRACK_FOLDERZ_ASpher/'
		for bin_file in os.listdir(bin_dir):
			if '.bin' in bin_file:
				bin_file = bin_dir+ bin_file
				x = joblib.load(bin_file)
				valid = False
				try:
					IMAGE_NAME = x[0]['imageName']
					print (IMAGE_NAME)
					thing_classes = x[1]
					imageWidth, imageHeight =x[0]['imgShape'][1], x[0]['imgShape'][0] #img.shape[1], img.shape[0]
					# print (IMAGE_PATH)
					IMAGE_PATH = IMAGE_FOLDER + IMAGE_NAME
					print (IMAGE_PATH)
					img = cv2.imread(IMAGE_PATH)
					print ("ere", IMAGE_PATH)
					print (img.shape)
					for num, box in enumerate(x[0]['predBoxes']):
						targetPixelX =  int((box[0]+box[2])/2)
						targetPixelY = int(box[3])

						aoi_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]

						output = reader.readtext(aoi_img, detail = 0, paragraph=True)
						print (output)
						outpput_str = " ".join(output)
						out_csv.write("{}\t{}\n".format(IMAGE_NAME, outpput_str))



						cv2.imwrite("aoi_ocr_15Aug/{}_{}.png".format(IMAGE_NAME,str(num)), aoi_img)

				except Exception as E:
					print (E)
					pass

out_csv.close()
