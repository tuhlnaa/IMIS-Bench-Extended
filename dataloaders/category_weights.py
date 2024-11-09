import pickle
from transformers import CLIPTextModel, AutoTokenizer, CLIPTextConfig
import torch
import re
import json  

import numpy as np 
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = CLIPTextModel(CLIPTextConfig())
tokenizer = AutoTokenizer.from_pretrained("clip-vit-base-patch32")
model = model.to(device)
  
# JSON文件路径  
file_path = r'E:\SAM-Med2Dv2\dataloaders\class_mapping.json'  
  
# 打开文件并读取内容  
with open(file_path, 'r', encoding='utf-8') as file:  
    json_data = json.load(file) 

ori_normal_removeLR = json_data['ori_mapping_remove_left_right']

categories = []   #204种
for key, value in ori_normal_removeLR.items():
    categories.append(value[1])

categories_cleaned = []
for category in set(categories):
    category = category.lower().replace('_', ' ').replace("-", " ")
    category = category.replace('left','').replace('right', '').strip()
    category = re.sub(r'\s+', ' ', category)
    categories_cleaned.append(category)

categories_cleaned.remove('background')
categories_cleaned = set(categories_cleaned)
categories = sorted(categories_cleaned)

print('cleaned categories:', len(categories))

index_to_category = {idx: cls for idx, cls in enumerate(categories)}
category_to_index = {cls: idx for idx, cls in enumerate(categories)}

templates = ['a {}.']
def gen_category_weights(model, tokenizer, device, categories, templates):
    category_embeds = []
    for category in categories:
        texts = [template.format(category) for template in templates]
        tokens = tokenizer(texts, padding=True, return_tensors="pt")

        for key in tokens.keys():
            tokens[key] = tokens[key].to(device)

        embeds = model(**tokens).pooler_output
        embeds = torch.nn.functional.normalize(embeds, dim=-1)

        if len(templates) > 1:
            embed = embeds.mean(dim=0)
            embed = torch.nn.functional.normalize(embed, dim=-1)
        else:
            embed = embeds[0]
        category_embeds.append(embed)
    return torch.stack(category_embeds, dim=-1)

category_weights = gen_category_weights(model, tokenizer, device, categories, templates)
category_weights = category_weights.detach().cpu().numpy()


# save_path = r'E:\SAM-Med2Dv2\dataloaders/categories_weight.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump([src_weights1, ori_normal_removeLR, category_to_index, index_to_category], f)



# concept_weights = r'E:\\SAM-Med2Dv2\\dataloaders\\categories_weight.pkl'
# with open(concept_weights, "rb") as f:
#     src_weights0, category_map0, category_to_index0, index_to_category0 = pickle.load(f)

# concept_weights = r'E:\SAM-Med2Dv2\dataloaders\categories_113.pkl'
# with open(concept_weights, "rb") as f:
#     src_weights1, category_map1 = pickle.load(f)

