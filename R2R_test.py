from data.env import R2RBatch
from data.data_utils import construct_instrs
from configs.config import args



args._parse(None)

## json file
train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=None, max_instr_len=args.max_instr_len
)
train_env = R2RBatch(train_instr_data,args.connectivity_dir,batch_size=args.batch_size,name = 'train')
val_env_names = ['val_train_seen', 'val_seen']
val_envs = {}
for split in val_env_names:
    val_instr_data = construct_instrs(
        args.anno_dir, args.dataset, [split], tokenizer=None, max_instr_len=args.max_instr_len
    )
    val_env = R2RBatch(
        val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        name=split)
    val_envs[split] = val_env
1111