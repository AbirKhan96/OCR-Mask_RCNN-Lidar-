import os
import pickle
import numpy as np
import laspy as lp
from tqdm import tqdm
from shapely import geometry

ROOT_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/"
MISSION_DIR = "2021-FEB-25_Mission2CGSML"

X = {}
Y = {}
Z = {}
bearing = {}
metadata = {}

orientation_folder_path = os.path.join(ROOT_FOLDER, MISSION_DIR, 'FOR_Orbit')
for track_folder in os.listdir(orientation_folder_path):
    if 'Track_' in track_folder:
        orientation_file_path = os.path.join(orientation_folder_path, track_folder, 'New Folder', 'External Orientation.csv')
        orientation_file = open (orientation_file_path, 'r')
        for line in orientation_file:
            cols = line.rstrip().split(';')
            image_id= cols[0].split('.')[0]
            X[image_id] = float(cols[2])
            Y[image_id] = float(cols[3])
            Z[image_id] = float(cols[4])
            bearing[image_id] = float(cols[6])
            metadata[image_id] = (X[image_id], Y[image_id], Z[image_id], bearing[image_id])




LIDAR_FOLDER = os.path.join(ROOT_FOLDER, MISSION_DIR,'FOR_Orbit','Las')
LIDAR =  os.listdir(LIDAR_FOLDER)


EXTENT_X = {}
EXTENT_Y = {}
EXTENT_Z = {}
LAS_IMAGE = {}

for las_file in LIDAR:

    LAS_FILE_PATH =  os.path.join(LIDAR_FOLDER, las_file)
    point_cloud=lp.file.File(LAS_FILE_PATH, mode="r")
    las_id = las_file
    LAS_IMAGE[las_id] = []


    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

    # Extent of Las file
    min_x_las, max_x_las = np.min(point_cloud.x), np.max(point_cloud.x)    # 270624.80000000005 270771.62000000005
    min_y_las, max_y_las = np.min(point_cloud.y), np.max(point_cloud.y)    # 2096102.2750000004 2096241.3900000004
    min_z_las, max_z_las = np.min(point_cloud.z), np.max(point_cloud.z)    # 9.440000000000005 53.2
    EXTENT_X[las_id] = (min_x_las, max_x_las)
    EXTENT_Y[las_id] = (min_y_las, max_y_las)
    EXTENT_Z[las_id] = (min_z_las, max_z_las)
    del point_cloud, points, colors
    


# Track_A_20210225_044653 Profiler.zfs_0 Sample LAs File Name

polygon_dict  = {}
for key in EXTENT_X:
    track_name = '_'.join(key.split('_')[0:2])
    las_id = key #key.split('.')[1].split('_')[-1]
    if track_name not in polygon_dict:
        polygon_dict[track_name] = []
    pointList = []


    pointList.append([EXTENT_X[key][0], EXTENT_Y[key][0]])
    pointList.append([EXTENT_X[key][0], EXTENT_Y[key][1]])
    pointList.append([EXTENT_X[key][1], EXTENT_Y[key][1]])
    pointList.append([EXTENT_X[key][1], EXTENT_Y[key][0]])
    pointList.append([EXTENT_X[key][0], EXTENT_Y[key][0]])
    poly = geometry.Polygon([p for p in pointList])
    polygon_dict[track_name].append((poly,las_id))
    




las_num_dict = {}

from shapely.geometry import Point


las_num_dict = {}
for image in X:
    point = Point(X[image], Y[image])

    buffer = point.buffer(40,1)
    for track_name in polygon_dict:
        if track_name not in las_num_dict:
            las_num_dict[track_name] = {}
        if track_name in image:
            for value in polygon_dict[track_name]:
                if buffer.overlaps(value[0]):
                    if image not in las_num_dict[track_name]:
                        las_num_dict[track_name][image] = []
                    las_num_dict[track_name][image].append(value[1])


print (las_num_dict)

ROOT_SAVE_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/AI_output/"


with open(ROOT_SAVE_FOLDER+'/{}_las_info.pickle'.format(MISSION_DIR), 'wb') as handle:
    pickle.dump(las_num_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ROOT_SAVE_FOLDER+'/{}_metadata.pickle'.format(MISSION_DIR), 'wb') as handle:
    pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)



