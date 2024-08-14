import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ImageLoader.ImageLoader import BratPro_Reader
from Architecture.ResNet import ResNetClassifier
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


device = 'cuda:0'
criterion = monai.losses.DiceLoss(include_background= False,sigmoid=True).to(device)
data = json.load(open('data_split.json'))


datadict_train = BratPro_Reader(data['train_dicts'],transform=train_transforms,segment=True)
datadict_val = BratPro_Reader(data['val_dicts'],transform=val_transforms,segment=True)

trainloader = DataLoader(datadict_train, batch_size=8, shuffle=True)
valloader = DataLoader(datadict_val, batch_size=1, shuffle=False)


# model = UNet(in_channels=1,out_channels=2,init_features=32).to(device)
# model_name = 'UNet'

# model = ResNetClassifier(in_channels=2,out_channels=4).to(device)

model = monai.networks.nets.UNet(spatial_dims=3, in_channels = 3,out_channels=3,channels =[16,32,64,128],strides=[2,2,2]).to(device)

model_name = 'UNet'+"/{:%d_%m_%y}/".format(datetime.now())
os.makedirs(f'./models/{model_name}',exist_ok=True)
os.makedirs(f'./results/{model_name}',exist_ok=True)
os.makedirs(f'./plots/{model_name}',exist_ok=True)

optimizer = optim.Adam(model.parameters(), lr = 1e-4, eps = 0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=10,min_lr = 2e-5,mode='min')

num_epochs = 200

train_losses = []
val_losses = []
best_loss = np.inf
min_epoch = 0

#############
# Train Loop 
#############

for epoch in range(0,num_epochs):
    torch.cuda.empty_cache()

    epoch_loss = 0

    model.train()

    with tqdm(range(len(trainloader))) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            err = 0
            image = data['input'].to(device)
            output = model.forward(image) 

            seg = data['seg'].to(device)

            err_dice = criterion(output,seg)

            err = err_dice

            model.zero_grad()

            err.backward()
            optimizer.step()
            pbar.set_postfix(Train_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            epoch_loss += err.item()
            del image
            del err


        train_losses.append([epoch_loss/len(trainloader)])
        print('Training Loss at epoch {} is : Total {}'.format(epoch,*train_losses[-1]))

    epoch_loss = 0
    model.eval()
    with tqdm(range(len(valloader))) as pbar:
        for i, data in zip(pbar, valloader):
            torch.cuda.empty_cache()
            err = 0
            with torch.no_grad():
                image = data['input'].to(device)
                output = model.forward(image) 

                seg = data['seg'].to(device)

                err_dice = criterion(output,seg)

                err = err_dice

                del image

            pbar.set_postfix(Val_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            epoch_loss += err.item()
            del err

        val_losses.append([epoch_loss/len(valloader)])
        print('Validation Loss at epoch {} is : Total {}'.format(epoch,*val_losses[-1]))
    
    scheduler.step(*val_losses[-1])

    if(epoch_loss<best_loss):
            best_loss = epoch_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'lr_scheduler_state_dict':scheduler.state_dict(),
            }, './models/'+model_name+'_state_dict_best_loss'+str(epoch)+'.pth')
    else:
            pass
            # early_stopping_counter-=1

    np.save('./results/'+model_name+'_loss.npy', [train_losses,val_losses])
    
    if(epoch%10==0):
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'lr_scheduler_state_dict':scheduler.state_dict(),
        }, './models/'+model_name+'_state_dict'+str(epoch)+'.pth')

torch.save({
    'epoch': epoch,
    'model_state_dict': model2.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'lr_scheduler_state_dict':scheduler.state_dict(),
    }, './models/'+model_name+'_state_dict'+str(epoch)+'.pth')