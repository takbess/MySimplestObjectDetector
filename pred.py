# train.py

print("\n start")

from pathlib import Path
import sys,os
import argparse

from old.TKM.OD import modeling
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from TKM.OD import metrics
import sys

FILE = Path(__file__).resolve() # このファイルの絶対パスを取得
ROOT = FILE.parents[0] # 実行ファイルの親ディレクトリの絶対パスを取得
# ROOT を絶対パスに追加して、ROOT以下のモジュールのインポートができるようにする
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT,Path.cwd())) # cwd からの相対パスに変換

from models import modeling
from utils import metrics
from utils import my_cocoapi

from utils.dataset import one_person_dataloader
import torchsummary
from time import time


def pred(args):
    model = modeling.SimpleObjectDetector()
    model = torch.load(args.weight)

    # show the model
    criterion = nn.L1Loss()
    model.eval()
    for i,data in enumerate(test_loader):
        image = data['image']
        bbox = data['bbox']
        image = image.cuda()
        bbox = bbox.cuda()

        pred = model(image)

        loss = criterion(pred,bbox)
        IoU = metrics.IoU(pred,bbox)
        print(f"loss:{loss} \nIoU:{IoU}")
        

        # show the first ones of each batch
        import numpy as np
        import matplotlib.pyplot as plt

        number = 0 # 0~batch_size

        image_ = image[number]
        image_ = image_.cpu()
        image_show = np.array(image_.transpose(0,2).transpose(0,1)*255,dtype=int)

        plt.imshow(image_show)

        bbox = pred[number].cpu()
        print("predicted bbox:",bbox)
        x,y,w,h = bbox.tolist()
        plt.plot([x,x+w,x+w,x,x],[y,y,y+h,y+h,y])
        print(IoU[number])
        print("predicted bbox:",x,y,w,h)
        plt.show()
        plt.clf()

        if i > 5:
            break

def parse_opt(known=False):
    parser = argparse.ArgumentParser(description=" I'm learning yolov5 ")
    parser.add_argument("--cfg",type=str, default="",help="model yaml path")

    return parser.parse_args()

def main(args):
    train()

if __name__ == "__main__":
    args = parse_opt()
    main(args)
    
