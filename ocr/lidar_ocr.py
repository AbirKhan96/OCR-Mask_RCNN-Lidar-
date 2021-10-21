import time
import os
import traceback
import pickle
import numpy as np
import math
import utm # for generating _lat_lon 
import pyproj
import plotly.express as px
import plotly.graph_objects as go
import laspy as lp
import json
from tqdm import tqdm
from pprint import pprint
from sklearn.neighbors import KDTree
from shapely.geometry import Point
from shapely.ops import transform
from utils import *
import math
import cv2
import easyocr
import os
reader = easyocr.Reader(["en",'hi'], gpu = True)


curr_time = str(time.time())
"""
altitude
np. set_printoptions(suppress=True)"""
np. set_printoptions(suppress=True)
def get_angle_from_pxl_y(pxl_y, center_y, fov_y):
    fov_center = fov_y/2
    if (pxl_y<center_y):
        angle = fov_center*(1-(pxl_y/center_y))
    else:
        angle = -(fov_center*((pxl_y-center_y)/center_y))
    return angle

def get_point_altitude(source_alt, depth, angle_deg):
    return source_alt + (depth*math.sin((math.pi/180)*angle_deg))


"""
visualisation
"""
# f=  open ('Shape_file.csv', 'w')
# f.close()
def decimate(population, factor=160):
    """ jumps by factor """
    return population[::factor]

def plot(points, beg_point, end_point, collision_points, ray_points):
    """all single points are np.ndarray of shape (1, 3)"""

    # plotting for all points is expensive. 
    # sub-sample them
    sampled_points = decimate(points, factor=100)

    colors = np.array([(0,0,0, 0.03)]*len(sampled_points))

    beg_point_color = [0, 255, 0, 1]
    end_point_color = [0, 0, 255, 1]
    col_point_colors = [[255, 0, 0, 1]]*len(collision_points)
    ray_points_col = [[240,128,128, 0.4]]*(len(ray_points)-2)

    _new_ps = np.concatenate([sampled_points, [beg_point[0], end_point[0]], collision_points, ray_points[1:-1]])
    _new_cs = np.concatenate([colors, [beg_point_color, end_point_color], col_point_colors, ray_points_col])


    fig = go.Figure(data =[go.Scatter3d(
                                    x = _new_ps[:,0],
                                    y = _new_ps[:,1],
                                    z = _new_ps[:,2],
                                    mode ='markers',
                                    marker = dict(size = 2, color=_new_cs))])
    fig.show()

"""
equidistant points between two 3d points
"""

def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)

def getEquidistantPoints(p1, p2, n):
    return np.array([(lerp(p1[0],p2[0],1./n*i), lerp(p1[1],p2[1],1./n*i), lerp(p1[2],p2[2],1./n*i)) for i in range(n+1)])

#X_value, Y_value = convert_lat_lon_to_XY(lon2, lat2, source_z)
def convert_lat_lon_to_XY(lon,lat,Z):
    lon_lat = Point(lon, lat, Z)
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:32643')
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    X_Y = transform(project, lon_lat)
    
    X = X_Y.x #[0]
    Y = X_Y.y #[1]
    return X,Y

# from metadata (meters)
def get_target_xyz(depth, source_x, source_y, source_z, target_pixel_x, target_pixel_y, 
                   image_width, image_height, fov_x, fov_y, center_bearing):
    

    # print (depth, source_x, source_y, source_z, target_pixel_x, target_pixel_y, 
    #                image_width, image_height, fov_x, fov_y, center_bearing)
    src_lon, src_lat = convert_XY_to_lat_lon(source_x, source_y, source_z)
    target_bearing = CalculateBearingfromFOV(target_pixel_x, image_width, fov_x, center_bearing)
    target_lat, target_lon = CalculatelatlonfromBearing(src_lat, src_lon, target_bearing, depth)

    (target_x, target_y), target_z = (convert_lat_lon_to_XY(target_lon, target_lat, source_z),
                        get_point_altitude(source_alt=source_z, depth=depth, angle_deg=get_angle_from_pxl_y(target_pxl_y, image_height/2, fov_y)))

    # print (target_bearing)
    return [target_x, target_y, target_z]

def first_colliding_point_idx_v1(kd_tree, beg_point: np.ndarray, end_point: np.ndarray, points, dist_between_points=0.05, min_collision_dist=0.5):
    """
    :@param beg_point: shape (1,3) - source location
    :@param end_point: shape (1,3) - point at maximum poissible distance from source
    :@param points: shape (1,n_points) - point cloud points
    """
    # n_equidistant_points = points_per_meter * total_num_of_meter
    n_equidistant_points = (1/dist_between_points) * int(np.linalg.norm(beg_point[0] - end_point[0]))
    #print('n_equidistant_points:', int(n_equidistant_points))
    equidistant_points = getEquidistantPoints(beg_point[0], end_point[0], int(n_equidistant_points))
    #print(equidistant_points)
    
    collision_idxs = []
    min_dist, min_idx = math.inf, -1
    for p in equidistant_points[20:]:
        dist, ind = kd_tree.query([p], k=1)
        if dist[0][0] < min_dist:
            min_dist = dist[0][0]
            min_idx = ind[0][0]
            # closer points to source will be @lower indices
            # and far away points to source will be @higher indices
            if min_dist<min_collision_dist:
                collision_idxs.append((min_idx, dist[0][0]))
            
    return collision_idxs, equidistant_points

LIDAR_FOLDER = "/home/itis/jaipur_new_las/Las_1.4_Road-2/"
LIDAR =  os.listdir(LIDAR_FOLDER)


IMAGE_NAME_FOLDER = "/home/itis/jaipur_new_las/PANO/Track_A.Ladybug/"
# name,Icon/href,Camera/longitude,Camera/latitude,Camera/altitude,Camera/heading,Camera/tilt,Camera/roll,ViewVolume/leftFov,ViewVolume/rightFov,ViewVolume/topFov,ViewVolume/bottomFov,ViewVolume/near,styleUrl,Point/coordinates,_id
metadata_csv =open('/home/itis/jaipur_new_las/AI_jaipur_Output/Track_A01.csv', 'r')
import os
import json

if os.path.exists('metadata.json'):
    with open('metadata.json', "r") as infile:
        metadata = json.load(infile)
else:
    metadata = {}
    line_num = 0
    from tqdm import tqdm
    for line in tqdm(metadata_csv):
        line_num = line_num +1
        if line_num ==1:
            continue
        data= line.split(',')
        lon = float(data[2])
        lat = float(data[3])
        Z = 441
        bearing = float(data[5])
        X,Y = convert_lat_lon_to_XY(lon,lat,Z)
        # print (X,Y, data[-1])
        metadata[data[-1].strip()] = (X,Y,bearing)

    print (metadata)

    with open('metadata.json', "w") as outfile:
        json.dump(metadata, outfile)
# image id is Image_name.split('.')[0]
# metadata.keys() Image id of all images of all tracks
# las_num_dict.keys() Image id of all images in all tracks

# pixel_path_dir = 

# pixel_path = "/home/itis/jaipur_new_las/PANO/test_AI_output/pixel_info_jaipur_test.csv"
# pixel_path = "error.txt"
# pixel_file = open (pixel_path, 'r')

# # Track_A_20210225_044653 Profiler.zfs_0 Sample LAs File Name
# Track_dict = dict()
# for line in pixel_file:
#     ROOT_SAVE_FOLDER, BIN_FOLDER, IMAGE_NAME, IMAGE_WIDTH, IMAGE_HEIGHT, ASSET, point_str = line.split('\t')
#     if IMAGE_FOLDER not in Track_dict:
#         Track_dict[IMAGE_FOLDER] = []
#     else:
#         Track_dict[IMAGE_FOLDER].append(line)

# pixel_file.close()

# print (Track_dict.keys())
# print (las_num_dict.keys())

LIDAR_LIST = os.listdir(LIDAR_FOLDER)
for las_index in range(0,len(LIDAR_LIST),3):
    las_loaded = {}
    point_cloud = []
    points_stack = []
    las_list = LIDAR_LIST[las_index:las_index+3]
    for las in las_list:
        if not las in las_loaded:
            print (las)
            LAS_FILE_PATH =  os.path.join(LIDAR_FOLDER, las)
            point_cloud1=lp.file.File(LAS_FILE_PATH, mode="r")
            point_cloud.append(point_cloud1)
            points1 = np.vstack((point_cloud1.x, point_cloud1.y, point_cloud1.z)).transpose()
            min_x_las, max_x_las = np.min(point_cloud1.x), np.max(point_cloud1.x)    # 270624.80000000005 270771.62000000005
            min_y_las, max_y_las = np.min(point_cloud1.y), np.max(point_cloud1.y)    # 2096102.2750000004 2096241.3900000004
            min_z_las, max_z_las = np.min(point_cloud1.z), np.max(point_cloud1.z)    # 9.440000000000005 53.2
            print (points1.shape)
            print (min_x_las, max_x_las)
            print (min_y_las, max_y_las)
            print (min_z_las, max_z_las)
            points_stack.append(points1)
            las_loaded[las] = 1
    print ("here")
    points = np.concatenate(points_stack)
    print (points.shape)    
    kdtree = KDTree(points)
    all_hits = []
    image_hit = []
    car_loc = []
    line_count = 0

    pixel_path = "Pixel_info_OCR_shop.csv"
    # pixel_path = "test_pixel.csv"
    # pixel_path = 'error.txt'
    pixel_file = open (pixel_path, 'r')

    for line in tqdm(pixel_file):
        print (line)
        try:
            ROOT_SAVE_FOLDER, BIN_FOLDER, MISSION_DIR,  IMAGE_NAME, IMAGE_WIDTH, IMAGE_HEIGHT, ASSET, box_str, point_str = line.split('\t')
            point_str = point_str[:-1]
            if point_str == "":
                continue
            image_id = IMAGE_NAME.split('.')[0].split('-')[-1]
            # COLLISION POINTS ACCUMULATOR
            collision_points = dict()
            # image_id = IMAGE_NAME.split('.')[0]
            # image metadata
            image_width = int(IMAGE_WIDTH) # x pixel max
            image_height = int(IMAGE_HEIGHT) # y pixel max
            # 360 -20.90897959 -180
            fov_x = 360 # around x pixel (degrees)
            fov_y = 180 # around y pixel (degrees)

            # lidar source (meters)
            source_x = float(metadata[str(int(image_id)+1)][0])
            source_y = float(metadata[str(int(image_id)+1)][1])
            source_z = float(440.0)
            center_bearing = float(metadata[str(int(image_id)+1)][2])
            present_car_loc = [source_x, source_y, source_z]
            car_loc.append(present_car_loc)
            print (source_x, source_y, source_z, center_bearing)
            print ("Above is the the sources information")
            hits = []
            random_mask_pxls = []
            point_str = point_str[:-1]
            print ("point_str", point_str)
            print ("box_str", box_str)
            for loc, point in enumerate(point_str.split(';')[0:1]):

                target_pxl_x = int(point.split(',')[1])
                target_pxl_y = int(point.split(',')[0])
                if target_pxl_x > image_width/2:
                    depth = 20
                else:
                    depth =  15
                dist_between_points = 0.05 # for points made on line from lidar source to point at `depth`
                min_collision_dist = 0.2 # min required distance between a point cloud and equidistant points on line
                beg_point = [source_x, source_y, source_z] # from metadata (meters)
                end_point = get_target_xyz(depth, source_x, source_y, source_z, target_pxl_x, target_pxl_y, 
                                image_width, image_height, fov_x, fov_y, center_bearing) # heavily relies on metadata (meters)
                print ("End Point :", end_point)
                collision_idxs, equidistant_points = first_colliding_point_idx_v1(kd_tree=kdtree, 
                                    beg_point=np.array([beg_point]), 
                                    end_point=np.array([end_point]), 
                                    points=points,
                                    dist_between_points=dist_between_points,
                                    min_collision_dist=min_collision_dist)
                if len(collision_idxs) > 0:
                    hits.append(points[collision_idxs[0][0]])
                    try:
                        img = cv2.imread(IMAGE_NAME_FOLDER+IMAGE_NAME)
                        print (img)
                        box = box_str.split(',')
                        box = [int(math.floor(float(value))) for value in box]
                        print (int(box[1]),int(box[3]), int(box[0]),int(box[2]),) 
                        aoi_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]
                        cv2.imwrite('aoi/{}_{}.png'.format(IMAGE_NAME, loc), aoi_img)
                        output = reader.readtext(aoi_img, detail = 0, paragraph=True, contrast_ths =0.3)
                        print (output)
                        output_str = " ".join(output)
                    except Exception as E:
                        output_str = "Error"
                        print ("Error in Image AOI")
                        print (E)
                        traceback.print_exc()

                else:
                    pass#print('missed')
                # print (equidistant_points)
            # hits =np.median(hits, axis=0)
            # all_hits.append(hits)
            # np.savetxt("Shape_file.csv", hits, delimiter=',')
            print ("hits ho gaya :", hits)
            if len (hits) > 0:
                # with open("OCR_Depth_10_20_Jaipur_Equidistant_Points_{}.csv".format(curr_time), "a") as ef:
                #     np.savetxt(ef, equidistant_points, delimiter = ",", fmt='%f')
                to_write = '{},{},{},'.format(ASSET,IMAGE_NAME, output_str)
                for value in hits[0]:
                    to_write = to_write + str(value)+','
                to_write = to_write[:-1]
                # print(hits, "hit ho gaya congratulations of car location ", present_car_loc)            
                shp =  open("Test_OCR_Depth_10_20_Jaipur_ShapeFile_{}.csv".format(curr_time), "a", encoding = 'utf-8')
                shp.write(to_write+"\n")
                shp.close()
                line_count = line_count +1
                if line_count ==100000000:
                    break

        except Exception as E:
            print (E)
            traceback.print_exc()
            

