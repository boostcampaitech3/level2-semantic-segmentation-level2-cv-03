import os
from importlib import import_module
from lr_scheduler import LR_Scheduler
import numpy as np
import torch
import pandas as pd

from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import torch.nn as nn

from model import criterion_entrypoint # model 폴더 내 loss.py
from model import build_model
from dataset.transform import *
from dataset import CustomDataLoader
from utils import * # wnadb 관련 함수



def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, scheduler, wandb):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = -1
    
    wandb.watch(model)
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device=device, dtype=torch.float), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)

            #### poly ####
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
            wandb.write_train(mIoU, acc, loss)
        
        # scheduler.step()
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_mIoU, avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            # if avrg_loss < best_loss:
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                # best_loss = avrg_loss
                best_mIoU = val_mIoU
                save_model(model, saved_dir, epoch+1, best=True)
            
            # save_model(model, saved_dir, epoch+1)
            wandb.write_val(avrg_loss, val_mIoU)

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

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

class Trainer:
    def __init__(self, config, save_dir, train_path, val_path, val_every):
        self.num_classes = 11
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir if save_dir else "./saved"

        ## dataset
        self.dataset_path = "../data"
        self.train_path = self.dataset_path + train_path
        self.val_path = self.dataset_path + val_path

        ## validation 주기
        self.val_every = val_every

        ## save directory
        self.save_dir = increment_path(os.path.join(self.save_dir, config.exp_name))
        makedirs(self.save_dir)

        ## wandb
        increment_name = self.save_dir.split('/')[-1]
        self.wandb = Wandb(**config.wandb, name=increment_name, config=config)

        

    def trainer_train(self, config):
        '''
        config = config.train
        '''
        # model
        # model_module = getattr(import_module("model"), config.model.name)
        # model = model_module(num_classes=self.num_classes).to(self.device)

        model, preprocessing_fn = build_model(config.model)

        # transform 적용
        transform_module = getattr(import_module("dataset"), config.augmentation.name)
        train_transform = transform_module(val_flag = False, preprocessing_fn=preprocessing_fn)
        val_transform = transform_module(val_flag = True, preprocessing_fn=preprocessing_fn)

        # dataset 구성
        train_dataset = CustomDataLoader(data_dir=self.train_path, mode='train', transform=train_transform)
        val_dataset = CustomDataLoader(data_dir=self.val_path, mode='val', transform=val_transform)

        # dataloader
        train_loader = DataLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  **config.train_data_loader)
        val_loader = DataLoader(dataset=val_dataset,
                                  collate_fn=collate_fn,
                                  **config.val_data_loader)

        


        # loss
        loss_module = getattr(import_module("model"), criterion_entrypoint(config.loss.name))
        criterion = loss_module(**config.loss.args)
        # criterion = nn.CrossEntropyLoss()

        # metric
        opt_module = getattr(import_module("torch.optim"), config.optimizer.type)
        optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), **config.optimizer.args)

        # lr_scheduler
        # sch_module = getattr(import_module("torch.optim.lr_scheduler"), config.lr_scheduler.type)
        # scheduler = sch_module(optimizer, **config.lr_scheduler.args)

        # sch_module = getattr(import_module("lr_scheduler"), LR_Scheduler)
        scheduler = LR_Scheduler('poly', config.optimizer.args.lr, config.num_epochs ,len(train_loader))
        
        train(config.num_epochs, model, train_loader, val_loader, criterion, optimizer, self.save_dir, self.val_every, self.device, scheduler, self.wandb)

