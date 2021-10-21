import pickle


ROOT_SAVE_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/AI_output/"
MISSION_DIR = "2021-FEB-25_Mission2CGSML"

with open(ROOT_SAVE_FOLDER+'/{}_las_info.pickle'.format(MISSION_DIR), 'rb') as handle:
    las_num_dict = pickle.load(handle)

with open(ROOT_SAVE_FOLDER+'/{}_metadata.pickle'.format(MISSION_DIR), 'rb') as handle:
    metadata = pickle.load(handle)


print (las_num_dict)
# print(metadata)