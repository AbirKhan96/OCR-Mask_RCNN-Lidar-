from pathlib import Path

DATA_BASE = Path('store') / 'data'
MODEL_BASE = Path('store') / 'model'

MODEL_NAME = 'OCR'

class TrainConfig:

    max_iter = 10000
    base_lr = 0.00025
    batch_size_per_img = 256
    num_workers = 25  # os.cpu_count()
    ims_per_batch = 16

    
class DataPreparationConfig:
    # while training all are true
    proc_json_files = False
    rm_dups = False # todo: checks
    train_test_split = False
    labelme2coco = True # needed for kpis only if files train, test, holdout contensts change
    reg_datasets = True # needed for kpis

    
class DataConfig:

    # ======================================================================================================
    # beg: dataset details
    # ======================================================================================================
    class AllowedClassesDataset:

        data_type = "InstanceSegmentation"

        all_jsons_dir = str(DATA_BASE / 'Allowed_Classes' / 'all_jsons') + '/'
        split_dataset_dir = str(DATA_BASE / 'Allowed_Classes') + '/'

        train_test_split = 0.9
        test_hldt_split = 0.4

        to_shape = (9216, 4608)
        ALLOWED_CLASSES = [
                          'shop_name'
                          ]

        thing_classes = ['shop name', 'shop_name']
        ext = 'JPG'
    # ======================================================================================================
    # end: dataset details
    # ======================================================================================================



class ModelConfig:

    # ======================================================================================================
    # beg: model details
    # ======================================================================================================
    class Allowed_ClassesModel:

        zoo_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        save_dir = str(MODEL_BASE / 'Allowed_Classes') + '/'
    # ======================================================================================================
    # end: model details
    # ======================================================================================================
