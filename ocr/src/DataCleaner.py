import os, json
from tqdm import tqdm
import itertools 

# #dir_names = ['train', 'test']
# dir_names = ['all_jsons']
dir_names = ['/home/itis/Desktop/Work_Flow_OCR/src/store/data/Allowed_Classes/all_jsons']
ext = 'json'

def json_path_to_dic(json_path):
    dic = None
    with open(json_path, 'rb') as fp:
        dic = json.load(fp)
    return dic


def get_thing_class_distribution(dic, thing_class_counts):
    
    for shape_dic in dic['shapes']:
        if shape_dic['label'] not in thing_class_counts:
            thing_class_counts[shape_dic['label']] = 0
        thing_class_counts[shape_dic['label']] += 1

    return thing_class_counts

def compare_json_and_img_name(img_file_name, json_file_name, json_path):
    json_name = json_file_name.split('.')[0]
    img_name = img_file_name.split('.')[0]

    if json_name != img_name:
        print(f"[CRITICAL WARNING] {json_path} has different image name {img_file_name}")
    


thing_class_distribution = {}
for dir_name in dir_names:

    print(f'working on {dir_name} dir ...')
    for json_file_name in tqdm([f for f in os.listdir(dir_name) if f.split('.')[-1].lower() == ext.lower()]):
        try:
            path = f'{dir_name}/{json_file_name}'
            json_dic = json_path_to_dic(path)

            thing_class_distribution = get_thing_class_distribution(
                dic=json_dic, 
                thing_class_counts=thing_class_distribution)

            compare_json_and_img_name(
                img_file_name=json_dic['imagePath'], 
                json_file_name=json_file_name,
                json_path=path
                )
        except Exception as E:
            print ("#"*80)
            print (json_file_name)
            print (E)
            print ("#"*80)
            

sorted_classes = sorted(thing_class_distribution, key=thing_class_distribution.get)
for k in sorted_classes:
    print(k, ':' ,thing_class_distribution[k])

len(sorted_classes)


sorted_classes



