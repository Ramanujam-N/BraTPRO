import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ImageLoader.ImageLoader import BratPro_Reader,BratPro_Reader_1channel
from Architecture.ResNet import ResNetClassifier
from Architecture.UNet import UNet
# from Architecture.LossFunctions import DiceLoss
from Architecture.Transformations import ToTensor3D,RandomRotation3D
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import monai
import os
from datetime import datetime

train_transforms = transforms.Compose([
            RandomRotation3D([10,10]),
            ToTensor3D(True)])

val_transforms = transforms.Compose([ToTensor3D(True)])

multi_channel = False
device = 'cuda:1'
criterion = monai.losses.DiceLoss(include_background= False,sigmoid=True).to(device)
mse_criterion = nn.MSELoss()
data = json.load(open('data_split.json'))

if(multi_channel):
    datadict_train = BratPro_Reader(data['train_dicts'],transform=train_transforms,segment=True)
    datadict_val = BratPro_Reader(data['val_dicts'],transform=val_transforms,segment=True)
else:
    datadict_train = BratPro_Reader_1channel(data['train_dicts'],transform=train_transforms,segment=True)
    datadict_val = BratPro_Reader_1channel(data['val_dicts'],transform=val_transforms,segment=True)

trainloader = DataLoader(datadict_train, batch_size=8, shuffle=True)
valloader = DataLoader(datadict_val, batch_size=1, shuffle=False)


# model = UNet(in_channels=1,out_channels=3,init_features=32).to(device)
# model_name = 'UNet'

if(multi_channel):
    model1 = monai.networks.nets.UNet(spatial_dims=3, in_channels = 1,out_channels=3,channels =[16,32,64,128],strides=[2,2,2],num_res_units=2).to(device)
    model2 = monai.networks.nets.UNet(spatial_dims=3, in_channels = 1,out_channels=3,channels =[16,32,64,128],strides=[2,2,2],num_res_units=2).to(device)
else:
    model1 = monai.networks.nets.UNet(spatial_dims=3, in_channels = 1,out_channels=1,channels =[16,32,64,128],strides=[2,2,2],num_res_units=2).to(device)
    model2 = monai.networks.nets.UNet(spatial_dims=3, in_channels = 1,out_channels=1,channels =[16,32,64,128],strides=[2,2,2],num_res_units=2).to(device)


if(multi_channel):
    model_name = 'UNet_channel3'+"/{:%d_%m_%y}/".format(datetime.now())
else:
    model_name = 'UNet_channel1'+"/{:%d_%m_%y}/".format(datetime.now())
    
os.makedirs(f'./models/{model_name}',exist_ok=True)
os.makedirs(f'./results/{model_name}',exist_ok=True)
os.makedirs(f'./plots/{model_name}',exist_ok=True)

optimizer = optim.Adam(list(model1.parameters())+list(model2.parameters()), lr = 1e-4, eps = 0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=10,min_lr = 2e-5,mode='min')

num_epochs = 500

train_losses = []
val_losses = []
best_loss = np.inf
min_epoch = 0

#############
# Train Loop 
#############
early_stopping_counter=20

for epoch in range(0,num_epochs):
    if(early_stopping_counter==0):
        break
    torch.cuda.empty_cache()

    epoch_loss = 0

    model1.train()
    model2.train()

    with tqdm(range(len(trainloader))) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            
            image1 = data['input_base'].to(device)
            image2 = data['input_follow'].to(device)

            output1 = model1.forward(image1) 
            output2 = model2.forward(image2) 

            seg1 = data['base_seg'].to(device)
            seg2 = data['follow_seg'].to(device)

            err1 = criterion(output1,seg1)
            err2 = criterion(output2,seg2)
            err3 = mse_criterion(output1-output2,seg1-seg2)
            err = err1+err2+err3
            model1.zero_grad()
            model2.zero_grad()

            err.backward()
            optimizer.step()
            pbar.set_postfix(Train_Loss = {np.round(err1.cpu().detach().numpy().item(), 5),np.round(err2.cpu().detach().numpy().item(), 5),np.round(err3.cpu().detach().numpy().item(), 5)})
            pbar.update(0)
            epoch_loss += err.item()
            del image1
            del image2
            del err


        train_losses.append([epoch_loss/len(trainloader)])
        print('Training Loss at epoch {} is : Total {}'.format(epoch,*train_losses[-1]))

    epoch_loss = 0

    model1.eval()
    model2.eval()
    with tqdm(range(len(valloader))) as pbar:
        for i, data in zip(pbar, valloader):
            torch.cuda.empty_cache()
            err = 0
            with torch.no_grad():
                image1 = data['input_base'].to(device)
                image2 = data['input_follow'].to(device)

                output1 = model1.forward(image1) 
                output2 = model2.forward(image2) 

                seg1 = data['base_seg'].to(device)
                seg2 = data['follow_seg'].to(device)

                err1 = criterion(output1,seg1)
                err2 = criterion(output2,seg2)
                err3 = mse_criterion(output1-output2,seg1-seg2)
                err = err1 + err2 + err3

            pbar.set_postfix(Val_Loss = {np.round(err1.cpu().detach().numpy().item(), 5),np.round(err2.cpu().detach().numpy().item(), 5),np.round(err3.cpu().detach().numpy().item(), 5)})
            pbar.update(0)
            epoch_loss += err.item()
            del image1
            del image2
            del err

        val_losses.append([epoch_loss/len(valloader)])

        print('Validation Loss at epoch {} is : Total {} '.format(epoch,*val_losses[-1]))
    
    scheduler.step(*val_losses[-1])

    if(epoch_loss<best_loss):
            early_stopping_counter=20
            best_loss = epoch_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict_base': model1.state_dict(),
            'model_state_dict_follow': model2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'lr_scheduler_state_dict':scheduler.state_dict(),
            }, './models/'+model_name+'_state_dict_best_loss'+str(epoch)+'.pth')
    else:
            early_stopping_counter-=1

    np.save('./results/'+model_name+'_loss.npy', [train_losses,val_losses])
    
    if(epoch%10==0):
        torch.save({
        'epoch': epoch,
        'model_state_dict_base': model1.state_dict(),
        'model_state_dict_follow': model2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'lr_scheduler_state_dict':scheduler.state_dict(),
        }, './models/'+model_name+'_state_dict'+str(epoch)+'.pth')

torch.save({
    'epoch': epoch,
    'model_state_dict_base': model1.state_dict(),
    'model_state_dict_follow': model2.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'lr_scheduler_state_dict':scheduler.state_dict(),
    }, './models/'+model_name+'_state_dict'+str(epoch)+'.pth')