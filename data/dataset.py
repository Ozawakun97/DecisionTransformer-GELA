from torch.utils import data
import torch
import random
import torch.nn.functional as F
import h5py
import pickle
#from config import adject_dict_path,img_feat_path,TEXT_EMBEDDINGS_PATH 
from const import ACTION_END_IDX,ACTION_PAD_IDX

import pandas as pd
import json

class R2R_seq(data.Dataset):
    def __init__(self,args,data_type,device):
        data_dir = args.data_dir_template % data_type
        self.df = pd.read_json(data_dir,lines=True)
        self.device = device
        with open(args.adject_dict_path,'r') as f:
            self.adj_list = json.load(f)
        self.img_feat = h5py.File(args.img_feat_path)
        instr_dir = args.text_embeddings_template % data_type
        with open(instr_dir,'rb') as t:    
            self.instr_embeddings = pickle.load(t)
        self.max_path_cnt = self.df['path'].apply(len).max()
        self.max_choice = ACTION_PAD_IDX
        self.feat_size = 768
    def generate_path_from_actions(self,scan,start_path,start_view_idx,actions,choice_type):
        choice_list = self.adj_list[f"{scan}_{start_path}_{start_view_idx}"]
        paths = [start_path]
        for action in actions:
            if action == ACTION_END_IDX or action == ACTION_PAD_IDX :
                break
            new_path = choice_list[action]['nextViewpointId']
            paths.append(new_path)
            new_view_idx = choice_list[action]['absViewIndex']
            choice_list = self.adj_list[f"{scan}_{new_path}_{new_view_idx}"]
        return paths
    def step(self,scan,path_id,view_idx,action):
        choice_list = self.adj_list[f"{scan}_{path_id}_{view_idx}"]
        new_path = choice_list[action]['nextViewpointId']
        new_view_idx = choice_list[action]['absViewIndex']
        return new_path,new_view_idx
    def generate_path_from_logits(self,scan,start_path,start_view_idx,actions,choice_type):
        pass
    def get_img_feats_from_choice_list(self,choice_list):
        ret = torch.zeros(self.max_choice,self.feat_size)
        for i in range(len(choice_list)):
            scan,view_point,view_idx = choice_list[i].split("_")[0],choice_list[i].split("_")[1],int(choice_list[i].split("_")[2])
            key = f"{scan}_{view_point}"
            ret[i,:] = torch.from_numpy(self.img_feat[key][view_idx,:])
        return ret
    def get_instr_embeddings(self,instr):
        embeddings = self.instr_embeddings[instr]
        return embeddings
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        row =  self.df.iloc[index]
        ##todo: get the id
        scan = row['scan']
        view_idxs = row['viewidxs']
        path = row['path']
        step_cnt = len(row['action_seq'])
        instr_cnt = len(row['instructions'])
        j = random.randint(0, instr_cnt - 1)
        instr_embedding = torch.from_numpy(self.get_instr_embeddings(f"{index}_{j}"))
        step = torch.arange(1,step_cnt + 1)
        action_seq = torch.LongTensor(row['action_seq']) 
        padding_length = max(0,self.max_path_cnt - step_cnt)
        step = F.pad(action_seq, (0, padding_length), value=0)
        reward = torch.zeros(self.max_path_cnt)
        action_seq = F.pad(action_seq, (0, padding_length), value=ACTION_PAD_IDX)
        mask = action_seq != ACTION_PAD_IDX
        choice_list_seq = row['action_choices']
        obs_seq = torch.zeros(self.max_path_cnt,self.max_choice,self.feat_size)
        obs_type = torch.zeros(self.max_path_cnt,self.max_choice)
        for i in range(len(choice_list_seq)):
            obs_type[i][0] = 2
            choice_list = choice_list_seq[i]
            obs_type[i][:len(choice_list)] = 1
            img_obs = self.get_img_feats_from_choice_list(choice_list)
            obs_seq[i,:,:] = img_obs
        obs_seq = obs_seq * instr_embedding
        item = {
            "scan":scan,
            "path":path,
            "view_idxs":view_idxs,
            "instr_embedding":instr_embedding.to(self.device),
            "obs_seq":obs_seq.to(self.device),
            "obs_types":obs_type.to(self.device),
            "action_seq":action_seq.to(self.device),
            "step":step.to(self.device),
            "reward":reward.to(self.device),
            "mask":mask.to(self.device)
        }
        #choice_list = self.adj_list[f"{row['scan']}_{path[i]}_{view_idx}"]
        return item#scan,path,view_idxs,obs_seq,action_seq,mask
def r2r_seq_collate_fn(batch):
    scans = [item['scan'] for item in batch]
    path = [item['path'] for item in batch]
    view_idxs = [item['view_idxs'] for item in batch]
    instr_embeddings = [item["instr_embedding"] for item in batch]
    obs_seqs = [item['obs_seq'] for item in batch]
    obs_types = [item['obs_types'] for item in batch]
    action_seqs = [item['action_seq'] for item in batch]
    steps = [item['step'] for item in batch]
    rewards = [item['reward'] for item in batch]
    masks = [item['mask'] for item in batch]
    return {'scans': scans, 'paths': path,
            'view_idxs':view_idxs,
            'instr_embeddings':torch.stack(instr_embeddings),
            'obs_seqs':torch.stack(obs_seqs),
            'obs_types':torch.stack(obs_types),
            'action_seqs':torch.stack(action_seqs),
            'steps':torch.stack(steps),
            'rewards':torch.stack(rewards),
            'masks':torch.stack(masks)
            }