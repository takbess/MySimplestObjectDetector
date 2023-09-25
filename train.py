# train.py

# print("\n start")

from pathlib import Path
import sys
import os
import argparse
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

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

# create one person dataset
# already made during parepare_dataset.sh
# from utils import prepare_one_person_dataset

# data_loader
from utils.dataset import data_loader

from models import modeling
from utils import metrics
from utils import my_cocoapi

import torchsummary
import time 


def train(cfg):
    # hyper parameter
    loss_index = 0
    epoch_num = int(cfg.epoch_num)
    batch_size = int(cfg.batch_size)
    IMAGE_SIZE= eval(cfg.IMAGE_SIZE)
    train_loader = data_loader('train', batch_size, IMAGE_SIZE)
    test_loader = data_loader('test', batch_size, IMAGE_SIZE)

    model = modeling.SimpleObjectDetector()
    # check model
    model.cuda()
    torchsummary.summary(model,(3,32,32),device="cuda")
    
    # mode
    show_progress = True

    lr = float(cfg.optimizer.lr)
    momentum = float(cfg.optimizer.momentum)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)

    # loss
    criterion = nn.L1Loss()    
    if torch.cuda.is_available():
        criterion.cuda()

    model.train()
    start = time.time()
    for epoch in range(1,epoch_num+1):
        sum_loss = 0
        sum_IoU = 0

        model.train()

        # 1 epoch training
        for data in train_loader:
            image = data['image']
            bbox = data['bbox']
            image_id = data['image_id']
            if __debug__:
                for img_id,img,bbx in zip(image_id,image,bbox):
                    import numpy as np
                    img = np.array(img.cpu())
                    img = img.transpose(1,2,0)
                    bbx = np.array(bbx.cpu())
                    my_cocoapi.save_image_w1bbox(img,bbx,f'images/{img_id}_train_loader.jpg')

            image = image.cuda()
            bbox = bbox.cuda()

            pred = model(image)
            # log.info(pred[0].tolist())

            loss = criterion(bbox,pred)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_IoU += metrics.IoU(bbox,pred).sum()

        # show_progress
        if show_progress:
            log.info("epoch: {}, mean_loss: {}, mean_IoU: {}, elapsed_time: {}".format(epoch, sum_loss/(len(train_loader)*batch_size),
                                                                    sum_IoU/(len(train_loader)*batch_size),
                                                                    time.time() - start  ))
        
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
                log.info("loss not implemented error")
                sys.exit(1)

            sum_loss += loss.item()
            sum_IoU += metrics.IoU(bbox,pred).sum()

        if show_progress:
            log.info("epoch: {}, mean_loss: {}, mean_IoU: {}, elapsed_time: {}".format(epoch, sum_loss/(len(train_loader)*batch_size),
                                                                    sum_IoU/(len(train_loader)*batch_size),
                                                                    time.time() - start  )) 
    
    log.info("The end of the training")
    return model

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    log.info("cfg is the following:")
    log.info(cfg)

    model = train(cfg)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    save_path = f"{output_dir}/model_final.pth"
    torch.save(model.state_dict,save_path)
    log.info(f"the final model.state is saved: {save_path}.")


if __name__ == "__main__":
    main()
