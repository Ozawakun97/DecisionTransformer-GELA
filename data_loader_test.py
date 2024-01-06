from data.dataset import R2R_seq,r2r_seq_collate_fn
from data.env import R2RBatch
from data.data_utils import construct_instrs
from torch.utils.data import DataLoader
from configs.config import args
args._parse(None)
#prepare r2r data
jsonl_path = r'/home/ubuntu/henny/r2r_seq/dataset/R2R_train_seq.jsonl'
data_set = R2R_seq(jsonl_path)
data_loader = DataLoader(data_set,batch_size=args.batch_size,num_workers=args.num_workers,collate_fn=r2r_seq_collate_fn)
#r2r env
train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=None, max_instr_len=args.max_instr_len
)
train_env = R2RBatch(train_instr_data,args.connectivity_dir,batch_size=args.batch_size,name = 'train')
for data in data_loader:
    test_scan = data['scans'][0]
    start_path = data['paths'][0][0]
    view_idx = data['view_idxs'][0][0]
    path_gt = data['paths'][0]
    action = data['action_seqs'][0].cpu().tolist()
    path_pred = data_set.generate_path_from_actions(test_scan,start_path,view_idx,action)
    scores = train_env._eval_item(test_scan,path_pred,path_gt)


