from TKM.OD import modeling
import torch
from torch.utils.data import DataLoader

# datasets.yaml loaded
import yaml,json
with open("datasets.yaml","r") as f:
    datasets_cfg = yaml.safe_load(f)

# mscoco.one_person dataset
one_person_datasets_cfg = datasets_cfg['mscoco']['one_person']
json_path = one_person_datasets_cfg['annotation']
img_dir = one_person_datasets_cfg['image_dir']

# dataset[i]['image'] = np.array([]) 3rd order tensor
# dataset[i]['bbox'] = (x,y,w,h)
IMAGE_SIZE=(32,32)
RESIZE=True

from pycocotools.coco import COCO
import os
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

coco = COCO(json_path)

with open(json_path,"r") as f:
    anns = json.load(f)

image_ids = [image['id'] for image in anns['images']]
dataset = list()
# i,image_ids, anns['images'] is in the same order
for i,id in tqdm(enumerate(image_ids)):
    data = dict()
    
    # dataset['image']
    file_name = anns['images'][i]['file_name']
    file_path = os.path.join(img_dir,file_name)
    image = Image.open(file_path)
    if RESIZE:
        original_image_size = image.size
        image = image.resize(IMAGE_SIZE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0,0,0],[1,1,1])
    ])
    # Some images are gray scale
    if image.mode == 'L':
        image = image.convert("RGB")
    image = transform(image)
    
    data['image'] = image #torch.tensor(image,dtype=torch.float).transpose(0,2).transpose(1,2)

    # dataset['bbox']
    AnnIds = coco.getAnnIds(id,catIds=1)
    data['bbox'] = torch.tensor(coco.loadAnns(AnnIds)[0]['bbox'],dtype=torch.float)
    if RESIZE:
        data['bbox'][0] = data['bbox'][0] * IMAGE_SIZE[1] / original_image_size[1] # なぜかx,y が逆、、、
        data['bbox'][1] = data['bbox'][1] * IMAGE_SIZE[0] / original_image_size[0]
        data['bbox'][2] = data['bbox'][2] * IMAGE_SIZE[1] / original_image_size[1]
        data['bbox'][3] = data['bbox'][3] * IMAGE_SIZE[0] / original_image_size[0]
    dataset.append(data)

# check data['image','bbox']
CHECK_DATA=False

if CHECK_DATA:
    import numpy as np
    import matplotlib.pyplot as plt

    image_show = np.array(data['image'].transpose(0,1).transpose(0,2),dtype=int) # (M,N,3) format
    image_show = image_show[:,:,[2,1,0]] # RGB->BGR 
    plt.imshow(image_show)

    # show bbox
    x,y,w,h = data['bbox']
    plt.plot([x,x+w,x+w,x,x],[y,y,y+h,y+h,y],color='black')

perm = torch.randperm(len(dataset))
train_num = int(len(dataset)*0.8)
train_dataset = torch.utils.data.Subset(dataset,indices=perm[:train_num])
test_dataset = torch.utils.data.Subset(dataset,indices=perm[train_num:])


batch_size=4
train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=True)