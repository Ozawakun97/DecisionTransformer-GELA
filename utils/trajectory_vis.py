import os
import pandas as pd
import json
import math
import shutil
ANGLE_INC = math.radians(30)

def convert_heading_to_idx(headings):
    return [round(heading / ANGLE_INC) % 12 + 12 for heading in headings]
def write_txt(path,str_list):
    with open(path,'w') as f:
        for str in str_list:
            f.write(f"{str}\n")

json_path = r'/home/ubuntu/henny/VLN-GELA/datasets/R2R/annotations/GELR2R/train_gel.jsonl'
example_base_dir = r'/home/ubuntu/henny/VLN-GELA/datasets/R2R/examples'
img_base_dir = r'/home/ubuntu/henny/Matterport3DSimulator/pre_compute_imgs'
#adj_list = pd.read_json(adj_list_json)
df = pd.read_json(json_path,lines=True)
df = df.head()
#df['viewidx'] = df['headings'].apply(convert_heading_to_idx)
#df['img_id'] = df.apply(lambda row: [f"{row['scan']}_{path}_{id}" for path, id in zip(row['path'], row['viewidx'])], axis=1)

for index, row in df.iterrows():
    example_root_dir = os.path.join(example_base_dir,row['scan'])
    os.makedirs(example_root_dir,exist_ok=True)
    txt_path = os.path.join(example_root_dir,'instructions.txt')
    instructions = row['instructions']
    write_txt(txt_path,instructions)
    for i,viewpointid in enumerate(row['path']):
        view_idx = row['path_viewindex'][i]

        path = "{}/{}/{}.jpg".format(row['scan'],viewpointid,view_idx)
        jpg_path = os.path.join(img_base_dir,path)
        new_path = os.path.join(example_root_dir,f"{i}.jpg")
        shutil.copyfile(jpg_path,new_path)
