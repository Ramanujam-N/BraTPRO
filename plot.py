import matplotlib.pyplot as plt
import numpy as np
import os
model_name = 'ResNet'
model_name = 'UNet_ResNet/5_8_23/'
os.makedirs(f'./plots/{model_name}',exist_ok=True)

path = './results/'+model_name+'_loss.npy'
def plot_loss_curves(path):
    x = np.load(path)
    plt.title('Dice + CE Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(x[0,:])
    plt.plot(x[1,:])
    plt.legend(['train','val'])
    plt.savefig('./plots/'+model_name+'_loss.png')
    plt.show()

plot_loss_curves(path)