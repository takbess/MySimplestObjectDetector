import yaml,json
import torch
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torch

# parameter
DATASET="PascalVOC_one_person"

# datasets.yaml loaded
with open("data/datasets.yaml","r") as f:
    datasets_cfg = yaml.safe_load(f)

RESIZE=True
def get_one_bbox_dataset(json_path,img_dir,IMAGE_SIZE,split):
    # json_path is in coco format, and one annotation for one image is expected
    # dataset[i]['image_id'] = int
    # dataset[i]['image'] = np.array([]) 3rd order tensor
    # dataset[i]['bbox'] = (x,y,w,h)

    coco = COCO(json_path)

    with open(json_path,"r") as f:
        anns = json.load(f)

    image_ids = [image['id'] for image in anns['images']]
    dataset = list()
    # i,image_ids, anns['images'] is in the same order
    for i,id in tqdm(enumerate(image_ids)):
        data = dict()
        
        # image
        file_name = anns['images'][i]['file_name']
        file_path = os.path.join(img_dir,file_name)
        image = Image.open(file_path)

        # bbox
        catId = coco.getCatIds('person')[0]
        AnnIds = coco.getAnnIds(id,catIds=catId)
        bbox = torch.tensor(coco.loadAnns(AnnIds)[0]['bbox'],dtype=torch.float)

        # transform前のimage_wbbox を表示
        # if __debug__:
        #     from utils import my_cocoapi
        #     import numpy as np
        #     img = np.array(image)
        #     bbx = np.array(bbox.cpu())
        #     my_cocoapi.save_image_w1bbox(img,bbx,f'images/{id}_{split}_before_trans.jpg')

        # dataset['image_id']
        data['image_id'] = id

        # dataset['image']
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
        import copy 
        data['bbox'] = copy.deepcopy(bbox)
        if RESIZE:
            data['bbox'][0] = bbox[0] * IMAGE_SIZE[1] / original_image_size[0] # なぜかx,y が逆、、、
            data['bbox'][1] = bbox[1] * IMAGE_SIZE[0] / original_image_size[1]
            data['bbox'][2] = bbox[2] * IMAGE_SIZE[1] / original_image_size[0]
            data['bbox'][3] = bbox[3] * IMAGE_SIZE[0] / original_image_size[1]
        
        dataset.append(data)

        # if __debug__:
        #     from utils import my_cocoapi
        #     import numpy as np
        #     img = np.array(data['image'])
        #     img = img.transpose(1,2,0)
        #     bbx = np.array(data['bbox'])
        #     my_cocoapi.save_image_w1bbox(img,bbx,f'images/{id}_{split}_after_trans.jpg')

        # データ数を制限してデバッグしやすくする
        # if __debug__:
        #     if i >= 10:
        #         break

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

    return dataset 



# mscoco.one_person dataset

def data_loader(split,batch_size,IMAGE_SIZE):
    json_path = datasets_cfg[DATASET][split]['annotation']
    img_dir = datasets_cfg[DATASET][split]['image_dir']

    dataset = get_one_bbox_dataset(json_path,img_dir,IMAGE_SIZE,split)
    data_loader= DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader