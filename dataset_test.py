from data.dataset import R2R_seq
import h5py
import pickle

jsonl_path = r'/home/ubuntu/henny/babywalk/tasks/R2R/formatted_data/R2R_train_seq.jsonl'

#data_set = R2R_seq(jsonl_path)

#img_feat = h5py.File("/home/ubuntu/henny/VLN-GELA/datasets/R2R/features/pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5", 'r')
with open("/home/ubuntu/henny/babywalk/tasks/R2R/instruction_embedding.pkl", 'rb') as file:
    loaded_data = pickle.load(file)
111