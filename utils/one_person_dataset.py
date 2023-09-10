import yaml
import json
import copy
import numpy as np
import os
from utils import my_cocoapi
from pycocotools.coco import COCO

DATASET="PascalVOC"

def create_one_person_dataset(json_path):
    COUNT = 1
    
    with open(json_path,"r") as f:
        anns = json.load(f)
    
    coco = COCO(json_path)
    person_id = coco.getCatIds(['person'])[0]
    
    # input anns
    # output anns with only one person
    # get images with only one person
    image_ids = [image['id'] for image in anns['images']]
    image_id_2_index = {image_id:i for i,image_id in enumerate(image_ids)}

    # 各 anns['annotations'] の person number を countする
    person_numbers = [0]*len(anns['images'])
    for ann in anns['annotations']:
        if ann['category_id'] == person_id: # personカテゴリなら
            person_numbers[image_id_2_index[ann['image_id']]] +=1

    person_numbers = np.array(person_numbers)
    ones = ( person_numbers==COUNT )

    image_ids = np.array(image_ids)
    ones_ids = image_ids[ones] # これが person 一人しか映っていない画像のid
    ones_ids = set(ones_ids)
    # ones_ids の画像のannotationだけを保存して使いやすくしておく。

    anns_new = copy.deepcopy(anns)
    anns_new['annotations'] = []
    for ann in anns['annotations']:
        if ann['image_id'] in ones_ids and ann['category_id'] == person_id:
            anns_new['annotations'].append(ann)

    anns_new['images'] = []
    for image in anns['images']:
        if image['id'] in ones_ids:
            anns_new['images'].append(image)

    # save
    save_path=json_path.replace(".json","_one_person.json")
    with open(save_path,"w") as f:
        json.dump(anns_new,f)
    print(f"one person dataset saved :{save_path}")
    print(f"the number of images: {len(anns_new['images'])}")
    print(f"the number of annotations: {len(anns_new['annotations'])}")

    return save_path


with open("data/datasets.yaml","r") as f:
    datasets = yaml.safe_load(f)

for split in ["train","test"]:
    json_path = datasets[DATASET][split]['annotation']
    save_path = create_one_person_dataset(json_path)

    img_dir = datasets[DATASET][split]['image_dir']


    # check images
    save_dir = os.path.join(img_dir,"original")
    try: os.makedirs(save_dir)
    except:
        print("folder already exists: ",save_dir)
    my_cocoapi.show_anns(json_path,img_dir,save_dir,NUM=10,use_random=False)
    print("example images for json: ",json_path," are saved in ", save_dir)

    save_dir = os.path.join(img_dir,"one_person")
    try:
        os.makedirs(save_dir)
    except:
        print("folder already exists: ",save_dir)
    my_cocoapi.show_anns(save_path,img_dir,save_dir,NUM=10,use_random=False)
    print("example images for json: ",save_path," are saved in ",save_dir)

    # 画像を実際に確認してみたら、複数人映っているのにGT bbox が1個だけのものがいくつかあった、、、。
    # 軽く見た感じ、８割くらいはいい感じの one person dataset になっていそう。


print("end")