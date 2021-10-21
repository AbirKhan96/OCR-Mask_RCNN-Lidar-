
# cleaned using `utils/checknames.py`
from pipeline.train.det2.trainer import Det2Trainer
from config import TrainConfig, DataConfig, ModelConfig, DataPreparationConfig
import os, json, io, base64, cv2, glob
from pprint import pprint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import labelme
# # configure trainer
# trainer = Det2Trainer(
#   data=DataConfig.POC102Assets_JsonDataset,
#   model=ModelConfig.POC102Assets_JsonModel,
#   cfg=TrainConfig)

# configure trainer
trainer = Det2Trainer(
  data=DataConfig.AllowedClassesDataset,
  model=ModelConfig.Allowed_ClassesModel,
  cfg=TrainConfig)
THING_CLASSES = DataConfig.AllowedClassesDataset.thing_classes


# !pip install loguru
# !pip install joblib
# !pip install labelme


import os
import joblib
import numpy
import copy
from loguru import logger

def dump(obj, saveby='./temp.bin'):
    ret = joblib.dump(obj, saveby)
    logger.debug(f"saved {ret}")

def load(from_path):
    return joblib.load(from_path)
def load_bin(from_path):
    return joblib.load(from_path)

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

metadata = MetadataCatalog.get("dataset_train")
def get_seg_dict(predictor, on_im, save_dir, fname, THING_CLASSES):
    

    im = cv2.imread(on_im)
    # format at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    
    # print(len(outputs.keys())) #--> 1
    dic = outputs['instances'].__dict__
    
    pred_class_list =list(dic['_fields']['pred_classes'].cpu().numpy())
    print (pred_class_list)
    del dic['_fields']['pred_classes']
    dic['_fields']['pred_classes'] = []
    for obj in pred_class_list:
        dic['_fields']['pred_classes'].append(THING_CLASSES[obj])
        
    dic['_fields']['pred_classes'] = numpy.array(dic['_fields']['pred_classes'])
    v = Visualizer(im[:, :, ::-1],
            metadata=metadata, 
            scale=0.5, 
            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #plt.imshow(out.get_image()[:, :, ::-1])
    #plt.show()
    #plt.close()
    
    ## To save image please un comment below line.
    dic['_fields']['pred_classes']
    img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_dir+'/'+fname, img)
    
    #print(dic['_image_size']) #--> (4608, 9216)
    
    # print(dic['_fields'].keys()) #--> ['pred_boxes', 'scores', 'pred_classes', 'pred_masks']
    
    # print(dic['_fields']['pred_boxes'].tensor.cpu().detach().numpy())
    


    dic = {
        'imgShape': dic['_image_size'],
        'predClasses': dic['_fields']['pred_classes'],
        'predBoxes': dic['_fields']['pred_boxes'].tensor.cpu().detach().numpy(),
        #'boxScores': dic['_fields']['scores'].cpu().detach().numpy()
        'boxScoresdraw_instance_predictions': dic['_fields']['scores'].cpu().detach().numpy()
    }
    return dic



def get_im_seg_info(images_dir, predictor, save_dir, THING_CLASSES, ext='jpg'):
    
    #  vis_metadata = load_bin(from_path=vis_metadata)  # train metadata

    write_dir = save_dir + f"eval_{images_dir[:-1].split('/')[-1]}" + '/'
    logger.debug(f"saving segmented images in {write_dir}")
    os.makedirs(write_dir, exist_ok=True)

    # check..
    # logger.debug(vis_metadata.thing_classes)

    #list_of_dics = []
    for im_name in os.listdir(images_dir):
        if im_name.split('.')[-1].lower() == ext.lower():
            
            dic = get_seg_dict(
                predictor=predictor,
                on_im=images_dir+im_name,
                save_dir = save_dir,
                fname= im_name,
                THING_CLASSES=THING_CLASSES
            )
            
            #from pprint import pprint
            #pprint(dic_str)
            dic['imageName'] = im_name
            
            
            joblib.dump((copy.deepcopy(dic), THING_CLASSES), write_dir + f'{im_name}.segmentation_info.bin')
            
            #list_of_dics.append(dic)

    return None   


THING_CLASSES = DataConfig.AllowedClassesDataset.thing_classes


# In[4]:


from pipeline.eval.det2.get_model import GetTrained

predictor, cfg = (
        GetTrained(ModelConfig.Allowed_ClassesModel.__name__.replace("Model", ""))
        .fetch(thresh=0.7, cfg=True))
get_im_seg_info("/home/itis/Desktop/Work_Flow_OCR/src/store/data/Allowed_Classes/test/", predictor, "store/data/Allowed_Classes/prediction/", THING_CLASSES, ext='jpg')


