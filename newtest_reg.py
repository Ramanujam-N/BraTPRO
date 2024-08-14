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


def get_tp_fp_fn_tn(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    tn = np.sum((1 - y_true) * (1 - y_pred))
    return tp, fp, fn, tn



test_transforms = transforms.Compose([ToTensor3D(True)])


device = 'cuda:0'
criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = monai.losses.DiceLoss(include_background= False,sigmoid=True).to(device)
 
data = json.load(open('data_split.json'))


datadict_test = BratPro_Reader(data['test_dicts'][:],transform=test_transforms)

testloader = DataLoader(datadict_test, batch_size=1, shuffle=False)


model1 = monai.networks.nets.resnet18(pretrained = False,n_input_channels = 2, num_classes =4).to(device)
model2 = monai.networks.nets.UNet(spatial_dims=3, in_channels = 2,out_channels=2,channels =[16,32,64],strides=[2,2]).to(device)
model1.load_state_dict(torch.load('models/UNet_ResNet/5_8_23/_state_dict_best_loss6.pth')['model1_state_dict'])
model2.load_state_dict(torch.load('models/UNet_ResNet/5_8_23/_state_dict_best_loss6.pth')['model2_state_dict'])

model_name = 'ResNet'

test_loss= 0
test_f1 = [[],[],[],[]]
test_balancedacc = [[],[],[],[]]

predictions = []
gt = []
model1.eval()
model2.eval()
with tqdm(range(len(testloader))) as pbar:
    for i, data in zip(pbar, testloader):
        torch.cuda.empty_cache()
        err = 0
        with torch.no_grad():
            image = data['input'].to(device)
            output = model2.forward(image) 

            seg = data['seg'].to(device)
            label = data['gt'].to(device)

            output2 = model1.forward(output)
            err_dice = criterion2(output,seg)
            err_ce = criterion1(output2,label)

            err = err_dice + err_ce            

            predictions.append(torch.argmax(output2).cpu().item())
            gt.append(label.cpu().item())

            del image
            del label

        pbar.set_postfix(Val_Loss = np.round(err.cpu().detach().numpy().item(), 5))
        pbar.update(0)
        test_loss += err.item()
        del err

    print([test_loss/len(testloader)])

metrics = {}
predictions = np.array(predictions)
gt = np.array(gt)
for cls in range(0,4):
    cls_true = (gt == cls).astype(int)
    cls_pred = (predictions == cls).astype(int)

    tp, fp, fn, tn = get_tp_fp_fn_tn(cls_true, cls_pred)

    metrics[f'class_{cls}'] = {
    'tp': int(tp),
    'fp': int(fp),
    'fn': int(fn),
    'tn': int(tn),
    'balanced_accuracy': float(1/2 * (tp / (tp + fn) + tn / (tn + fp))),
    'f1_score': float(2 * tp / (2 * tp + fp + fn)),
    'true_positive_rate': float(tp / (tp + fn)),
    'true_negative_rate': float(tn / (tn + fp))}

print(metrics)