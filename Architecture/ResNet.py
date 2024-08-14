import torch
import torch.nn as nn

class ResNet3DBasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(output_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
    
class ResNet3D(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # self.layer5 = self._make_layer(512, 1024, 2, stride=2)
        # self.layer6 = self._make_layer(1024, 2048, 2, stride=2)

    def _make_layer(self, in_channels, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(output_channels),
            )
        
        layers = []
        layers.append(
            ResNet3DBasicBlock(
                self.in_channels,
                output_channels, 
                stride,
                downsample
            )
        )
        self.in_channels = output_channels

        for _ in range(1, blocks):
            layers.append(
                ResNet3DBasicBlock(
                    self.in_channels,
                    output_channels,
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)

        return x


class Classifier(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(input_channels // 2, input_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Linear(input_channels // 4, output_channels)

        self.fc = nn.Sequential()
        self.fc.add_module('layer1', self.layer1)
        self.fc.add_module('layer2', self.layer2)
        self.fc.add_module('layer3', self.layer3)

        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.encoder = ResNet3D(in_channels)
        self.classifier = Classifier(32768,out_channels) 

    def forward(self,x): 
        out = self.encoder(x)
        out = torch.flatten(out,1,-1)
        out = self.classifier(out)
        return out
    