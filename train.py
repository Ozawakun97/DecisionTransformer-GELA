from data.dataset import R2R_seq,r2r_seq_collate_fn
from data.env import R2RBatch
from data.data_utils import construct_instrs
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.decision_transformer import DecisionTransformer

import os
from tqdm import tqdm
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver,DecisionTransformerSaver, save_training_meta
from eval.eval_model import validate
from collections import defaultdict

from configs.config_01 import args
from const import ACTION_PAD_IDX

args._parse(None)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#training related tools
save_training_meta(args)
TB_LOGGER.create(os.path.join(args.output_dir, 'logs'))
pbar = tqdm(total=args.num_train_steps)
model_saver = DecisionTransformerSaver(os.path.join(args.output_dir, 'ckpts'),prefix="dst")
add_log_to_file(os.path.join(args.output_dir, 'logs', 'log.txt'))

#dataset
LOGGER.info(f"construct train dataset")
train_data_set = R2R_seq(args,"train",device)
train_data_loader = DataLoader(train_data_set,batch_size=args.batch_size,collate_fn=r2r_seq_collate_fn)
val_data = ['val_seen','val_unseen']
val_data_collection = {}
for val_data_type in val_data:
    val_data_set = R2R_seq(args,val_data_type,device)
    val_data_loader = DataLoader(val_data_set,batch_size=args.batch_size,collate_fn=r2r_seq_collate_fn)
    val_instr_data = construct_instrs(
        args.anno_dir, args.dataset, [val_data_type], tokenizer=None, max_instr_len=args.max_instr_len
    )
    val_env = R2RBatch(
        val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        name=val_data_type)
    val_data_collection[val_data_type] = {"dataset":val_data_set,"dataloader":val_data_loader,"env":val_env}
val_seen_best = defaultdict(float)
val_unseen_best = defaultdict(float)
LOGGER.info(f"initialize DecisionTransformer with {args.n_layer} layers and {args.n_head} heads")
#model
dst_model = DecisionTransformer(
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        max_length=args.max_lenth,
        max_ep_len=args.max_ep_len,
        hidden_size=args.hidden_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4*args.hidden_size,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout
)
dst_model = dst_model.to(device = device)
#dst_model = torch.compile(dst_model)
dst_model.train()
optimizer = torch.optim.AdamW(
        dst_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
)
LOGGER.info(f"set learning rate {args.learning_rate}")
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps+1)/args.warmup_steps, 1)
)
LOGGER.info(f"***** Running training with {args.gpu_nums} GPUs *****")
LOGGER.info("  Batch size = %d", args.batch_size)
LOGGER.info("  Num steps = %d", args.num_train_steps)
acc_loss = 0.0
step = 0
for _ in range(args.epoch_num):
    for batch in train_data_loader:
        # obs = batch['obs_seqs'].to(device)
        # mask = batch['masks'].to(device)
        # steps = batch['steps'].to(device)
        # returns = batch['rewards'].to(device)
        # act_seq_vec = F.one_hot(actions,ACTION_PAD_IDX + 1).to(torch.float32).to(device)
        actions = batch['action_seqs']
        action_preds = dst_model.get_action_pred(batch,device)#dst_model.forward_(obs, act_seq_vec, None, returns, steps, attention_mask=mask)
        optimizer.zero_grad()
        loss = F.cross_entropy(action_preds.view(-1,action_preds.shape[-1]),actions.view(-1),ignore_index=ACTION_PAD_IDX)
        acc_loss += loss.item()
        loss.backward()
        # learning rate scheduling
        lr_this_step = scheduler.get_last_lr()[0] #get_lr_sched(global_step, opts)
        TB_LOGGER.add_scalar('lr', lr_this_step, step)
        TB_LOGGER.add_scalar('entroy_loss', loss.item(), step)
        TB_LOGGER.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                        dst_model.parameters(), args.grad_norm
                    )
        TB_LOGGER.add_scalar('grad_norm', grad_norm, step)
        optimizer.step()
        scheduler.step()
        pbar.update(1)
        step += 1
        if (step + 1) % args.log_steps == 0:
            # monitor training throughput
            LOGGER.info(f'==============Step {step}===============')
            avg_loss =  acc_loss / args.log_steps
            acc_loss = 0.0
            LOGGER.info(f'from {step - args.log_steps} to {step} : average loss :{avg_loss}')
            LOGGER.info('===============================================')
        if (step + 1) % args.val_steps == 0:
            for val_type in val_data:
                LOGGER.info(f'------Step {step}: start {val_type}------')
                val_score = validate(dst_model,val_data_collection,val_type,device)
                for k, v in val_score.items():
                    TB_LOGGER.log_scalar_dict(
                    {f'{val_type}/{k}': v}
                    )
                    LOGGER.info(f'{val_type}/{k} : {v}')
                if 'unseen' in val_type:
                    val_unseen_best['nav_error'] = min(val_unseen_best['nav_error'],val_score['nav_error'])
                    val_unseen_best['sr'] = max(val_unseen_best['sr'],val_score['sr'])
                    val_unseen_best['spl'] = max(val_unseen_best['spl'],val_score['spl'])
                    val_unseen_best['cls'] = max(val_unseen_best['cls'],val_score['cls'])
                    val_unseen_best['SDTW'] = max(val_unseen_best['SDTW'],val_score['SDTW'])
                    for k,v in val_unseen_best.items():
                        if 'nav_error' in k:
                            continue
                        model_saver.save(dst_model,optimizer,scheduler,f"{val_type}_best_{k}",v)
                else:
                    val_seen_best['nav_error'] = min(val_seen_best['nav_error'],val_score['nav_error'])
                    val_seen_best['sr'] = max(val_seen_best['sr'],val_score['sr'])
                    val_seen_best['spl'] = max(val_seen_best['spl'],val_score['spl'])
                    val_seen_best['cls'] = max(val_seen_best['cls'],val_score['cls'])
                    val_seen_best['SDTW'] = max(val_seen_best['SDTW'],val_score['SDTW'])
                    for k,v in val_seen_best.items():
                        if 'nav_error' in k:
                            continue
                        model_saver.save(dst_model,optimizer,scheduler,f"{val_type}_best_{k}",v)
