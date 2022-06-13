import os
import random
import time
import json
import warnings
from dataset import CustomDataLoader
from dataset import BasicTransform
from lr_scheduler import LR_Scheduler 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import *

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

#!pip install albumentations==0.4.6
import albumentations as A
from albumentations.pytorch import ToTensorV2

# !pip install wandb
import wandb

import segmentation_models_pytorch as smp

from model import DiceCELoss


def seed_everything(random_seed):
    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# smp 라이브러리에서 모델 불러오는 함수
def build_model(decoder, encoder, encoder_weights):
    decoder = getattr(smp, decoder)
    model = decoder(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=11,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    return model, preprocessing_fn


# optimizer 불러오는 함수. 
def build_optimizer(model, optimizer, learning_rate, weight_decay=0,):
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9, weight_decay= weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(),
                                lr = learning_rate, weight_decay= weight_decay)
    # 사용하고 싶은 옵티마이저가 있으시면 위 형식처럼 추가해주시면 됩니다.
    return optimizer



def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, val_every, device, scheduler,saved_dir=None):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = -1
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            # images.to(device=device, dtype=torch.float) 를 했는데 원래 코드는 images.to(device)입니다.
            # 저는 dtype 에러가 발생해서 아래처럼 바꿔 넣었습니다
            images, masks = images.to(device=device, dtype=torch.float), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)

            #### poly ####
            # 이 부분은 적용하고 싶으신 scheduler에 따라서 바꿔주시면 됩니다. 
            # poly는 iter수 마다 lr을 계산하기 때문에 여기에 추가했습니다.
            # if (epoch+1)%2==0:
            scheduler(optimizer, step, epoch, best_mIoU, 0.94)
            ##############

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
            wandb.log({
                'train_mIoU':mIoU,
                'train_acc':acc,
                'train_loss':loss.item()
            })
        
        # scheduler.step()
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_mIoU, avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            # if avrg_loss < best_loss:
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                # print(f"Save model in {saved_dir}")
                # best_loss = avrg_loss
                best_mIoU = val_mIoU
                save_model(model, saved_dir, epoch+1, best=True)

            
            wandb.log({
                    'val_loss' : avrg_loss,
                    'val_mIoU' : val_mIoU,
                })

category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device=device, dtype=torch.float), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return mIoU, avrg_loss


def collate_fn(batch):
    return tuple(zip(*batch))


########### hyperparameter 초기값##############
hyperparameter_defaults = dict(
    batch_size = 16,
    learning_rate = 0.001,
    weight_decay = 4e-5,
    epochs = 10,
    optimizer = "sgd"
    )

## wandb
wandb = wandb.init(config=hyperparameter_defaults)
config = wandb.config

def main():
    ## dataset
    dataset_path = "../data"
    train_path = dataset_path + "/train.json"
    val_path = dataset_path + "/val.json"

    save_dir = "./saved"
    save_dir = increment_path(os.path.join(save_dir, f"wandb_sweep/{wandb.name}"))
    makedirs(save_dir)

    ## validation 주기
    val_every = 1

    # model load
    model, preprocessing_fn = build_model('DeepLabV3Plus', "tu-xception65", "imagenet")

    wandb.watch(model)

    # transform 적용
    train_transform = BasicTransform(val_flag = False, preprocessing_fn=preprocessing_fn)
    val_transform = BasicTransform(val_flag = True, preprocessing_fn=preprocessing_fn)

    # dataset 구성
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
    
    # dataloader
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                )
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=config.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                )

    # loss
    criterion = DiceCELoss()

    # metric
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

    # lr_scheduler
    # poly scheduler가 아닌 다른 scheduler를 사용하고 싶으시면 바꿔사용해 주세요!!
    scheduler = LR_Scheduler('poly', config.learning_rate, config.epochs ,len(train_loader))
    

    # train
    print("Start training...")
    seed_everything(21)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(config.epochs, model, train_loader, val_loader, criterion, optimizer, val_every, device, scheduler, saved_dir=save_dir)


if __name__ == '__main__':
   main()