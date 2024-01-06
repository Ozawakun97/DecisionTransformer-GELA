from utils.logger import LOGGER, TB_LOGGER
from const import ACTION_PAD_IDX
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
@torch.no_grad()
def validate(model, val_collection, setname,device):
    model.eval()
    val_data_set =  val_collection[setname]["dataset"]
    val_data_loader = val_collection[setname]["dataloader"]
    val_env = val_collection[setname]["env"]
    LOGGER.info(f"start running {setname} validation...")
    val_metrics = defaultdict(float)
    data_num = len(val_data_loader)
    for batch in val_data_loader:
        # obs = batch['obs_seqs']
        # actions = batch['action_seqs']
        # mask = batch['masks']
        # steps = batch['steps']
        # returns = batch['rewards']
        # act_seq_vec = F.one_hot(actions,ACTION_PAD_IDX+1).to(torch.float32)
        #action_preds = model.forward_(obs, act_seq_vec, None, returns, steps, attention_mask=mask)
        #action_preds = model.get_action_pred(batch,device)
        #action_preds[batch['obs_types'] == 0] = float('-inf')
        #action_seq = action_preds.max(dim=-1)[1] # batch seq 7
        instr_embeddings = batch['instr_embeddings']
        scans = batch['scans']
        paths = batch['paths']
        view_idxs = batch['view_idxs']
        batch_size = len(scans)
        metrics = defaultdict(list)
        for i in range(batch_size):
            test_scan = scans[i]
            path_gt = paths[i]
            instr_embedding = instr_embeddings[i]
            start_path = path_gt[0]
            view_idx = view_idxs[i][0]
            #action = action_seq[i].cpu().tolist()
            path_pred = evaluate_episode(val_data_set,model,instr_embedding,test_scan,start_path,view_idx,max_episode_len=val_data_set.max_path_cnt)
            #path_pred = None#val_data_set.generate_path_from_actions(test_scan,start_path,view_idx,action,batch['obs_types'][i])
            traj_scores = val_env._eval_item(test_scan,path_pred,path_gt)
            for k, v in traj_scores.items():
                metrics[k].append(v)
        batch_avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        } 
        for k, v in batch_avg_metrics.items():
            val_metrics[k] += v
    model.train()
    for k,_ in val_metrics.items():
        val_metrics[k] /= data_num
    return val_metrics

def evaluate_episode(dataset,model,instr_embedding,scan_id,start_path,start_view_idx,max_episode_len):
    device = instr_embedding.device
    rewards = torch.zeros(max_episode_len).to(device)
    masks = torch.zeros(max_episode_len).to(device)
    steps = torch.zeros(max_episode_len).to(device).to(torch.int64)
    actions = torch.full((max_episode_len,), ACTION_PAD_IDX).to(device)
    obs_seq = torch.zeros((max_episode_len,ACTION_PAD_IDX,instr_embedding.shape[-1])).to(device)
    path_id = start_path
    view_idx = start_view_idx
    path = [start_path]
    for i in range(max_episode_len):
        choice_list = dataset.adj_list[f"{scan_id}_{path_id}_{view_idx}"]
        choice = []
        for j in range(len(choice_list)):
            cand_view_idx = choice_list[j]['absViewIndex'] if choice_list[j]['absViewIndex'] != -1 else view_idx
            cand_view_point = choice_list[j]['nextViewpointId']
            choice.append(f"{scan_id}_{cand_view_point}_{cand_view_idx}")
        obs_img_feats = dataset.get_img_feats_from_choice_list(choice).to(device)
        obs = obs_img_feats * instr_embedding
        obs_seq[i] = obs
        masks[i] = 1
        steps[i] = i+1
        act_seq_vec = F.one_hot(actions,ACTION_PAD_IDX + 1).to(torch.float32).to(device)
        action_pred_logits = model.predict_action(obs_seq,rewards,act_seq_vec,steps,masks,i)
        action_pred_logits[len(choice_list):] = float('-inf')
        action_pred = action_pred_logits.max(dim=-1)[1].item()
        if action_pred == 0:
            return path
        new_path , new_view_idx = dataset.step(scan_id,path_id,view_idx,action_pred)
        path_id = new_path
        view_idx = new_view_idx
        actions[i] = action_pred
        path.append(new_path)
        #action = actions[:i+1]
    return path
