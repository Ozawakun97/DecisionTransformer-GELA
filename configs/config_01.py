# coding:utf8
import warnings
import torch as t
from const import ACTION_PAD_IDX

class DefaultConfig(object):
    gpu_nums = 1
    #data
    output_dir = r'exprs/exp01'
    anno_dir = r'dataset'
    data_dir_template = 'dataset/R2R_%s_seq.jsonl'
    dataset = 'r2r'
    connectivity_dir = r'dataset/connectivity'
    adject_dict_path = r'/home/ubuntu/henny/babywalk/simulator/total_adj_list.json'
    img_feat_path = r"feats/pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5"
    text_embeddings_path = r'feats/instr_embeddings.pkl'
    text_embeddings_template = 'feats/instr_%s_embeddings.pkl'
    batch_size = 64
    num_workers = 4
    max_instr_len = 60
    #model input
    state_dim = 768
    act_dim = ACTION_PAD_IDX + 1
    max_lenth = 10
    max_ep_len = 30
    #model
    hidden_size = 768
    n_layer = 12
    n_head = 12
    activation_function = 'relu'
    dropout = 0.1
    #training
    epoch_num = 1000
    learning_rate = 1e-4
    weight_decay = 1e-4
    warmup_steps = 10000
    log_steps = 200
    val_steps = 500
    num_eval_episodes = 100
    epochs = 10
    iters_per_epoch = 10000
    num_train_steps = 200000
    grad_norm = 5.0
    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        if kwargs:
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    warnings.warn("Warning: opt has not attribut %s" % k)
                setattr(self, k, v)
        
        #args.


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

args = DefaultConfig()
