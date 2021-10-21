#!/usr/bin/env python
# coding: utf-8

# In[1]:


# cleaned using `utils/checknames.py`
from pipeline.train.det2.trainer import Det2Trainer
from config import TrainConfig, DataConfig, ModelConfig

# configure trainer
trainer = Det2Trainer(
  data=DataConfig.AllowedClassesDataset,
  model=ModelConfig.Allowed_ClassesModel,
  cfg=TrainConfig)

class DataPreparationConfig:
    # while training all are true
    proc_json_files = True
    rm_dups = True # todo: checks
    train_test_split = True
    labelme2coco = True # needed for kpis only if files train, test, holdout contensts change
    reg_datasets = True # needed for kpis


# process the json files into trainable data
trainer.prepare_data(DataPreparationConfig)
# generates out directory for assets' model after training
# and evaluating


trainer.start(
    resume=True,
    train_dataset=("dataset_train",),
    test_dataset=("dataset_test",))


# In[1]:


# !pip install loguru
# !pip install joblib
# !pip install labelme

