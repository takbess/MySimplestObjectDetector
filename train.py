# train.py

# print("\n start")

from pathlib import Path
import sys
import os
import argparse

# from old.TKM.OD import modeling
import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
# from TKM.OD import metrics


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


def train():
    # hyper parameter
    loss_index = 0
    epoch_num = 10
    batch_size = 16 
    train_loader,test_loader = one_person_dataloader()

    model = modeling.SimpleObjectDetector()
    # check model
    model.cuda()
    torchsummary.summary(model,(3,32,32),device="cuda")
    
    # mode
    show_progress = True

    
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

    # loss
    criterion = nn.L1Loss()    
    if torch.cuda.is_available():
        criterion.cuda()

    model.train()
    start = time()
    for epoch in range(1,epoch_num+1):
        sum_loss = 0
        sum_IoU = 0

        model.train()

        # 1 epoch training
        for data in train_loader:
            image = data['image']
            bbox = data['bbox']
            image = image.cuda()
            bbox = bbox.cuda()

            pred = model(image)
            # print(pred[0].tolist())

            loss = criterion(bbox,pred)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_IoU += metrics.IoU(bbox,pred).sum()

        # show_progress
        if show_progress:
            print("epoch: {}, mean_loss: {}, mean_IoU: {}, elapsed_time: {}".format(epoch, sum_loss/(len(train_loader)*batch_size),
                                                                    sum_IoU/(len(train_loader)*batch_size),
                                                                    time() - start  ))
        
        # test
        model.eval()
        for data in test_loader:
            image = data['image']
            bbox = data['bbox']
            image = image.cuda()
            bbox = bbox.cuda()

            pred = model(image)

            if loss_index==0:
                loss = criterion(bbox,pred)
            elif loss_index==1:
                loss = criterion(bbox,pred) + metrics.IoU(bbox,pred).sum()
            elif loss_index==2:
                loss = metrics.IoU(bbox,pred).sum()
            else:
                print("loss not implemented error")
                sys.exit(1)

            sum_loss += loss.item()
            sum_IoU += metrics.IoU(bbox,pred).sum()

        if show_progress:
            print("epoch: {}, mean_loss: {}, mean_IoU: {}, elapsed_time: {}".format(epoch, sum_loss/(len(train_loader)*batch_size),
                                                                    sum_IoU/(len(train_loader)*batch_size),
                                                                    time() - start  )) 
    torch.save(model.state_dict,"output/model.pth")


def main(args):
    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" I'm learning yolov5 ")
    parser.add_argument("--cfg",type=str, default="",help="model yaml path")
    parser.add_argument("--epoch",default=10,help="epoch=10 in default")
    args = parser.parse_args()
    main(args)
