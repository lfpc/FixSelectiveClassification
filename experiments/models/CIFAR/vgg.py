'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.name = vgg_name

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_FCDropout(VGG):
    '''Took from https://github.com/chengyangfu/pytorch-vgg-cifar10'''
    def __init__(self, vgg_name, num_classes, drop_rate = 0.5):
        super().__init__(vgg_name, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

class VGG_11(VGG):
    def __init__(self, num_classes):
        super().__init__('VGG11',num_classes)
class VGG_13(VGG):
    def __init__(self, num_classes):
        super().__init__('VGG13',num_classes)
class VGG_16(VGG):
    def __init__(self, num_classes):
        super().__init__('VGG16',num_classes)
class VGG_19(VGG):
    def __init__(self, num_classes):
        super().__init__('VGG19',num_classes)

class VGG_16_FCDropout(VGG_FCDropout):
    def __init__(self, num_classes, drop_rate = 0.5):
        super().__init__('VGG16',num_classes, drop_rate)


from torch.nn import functional as F
class Conv2dSame(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class VGG_16_Dropout(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = Conv2dSame(3, 64, 3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_dropout = nn.Dropout(0.3)
        self.conv2 = Conv2dSame(64, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = Conv2dSame(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_dropout = nn.Dropout(0.4)
        self.conv4 = Conv2dSame(128, 128, 3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = Conv2dSame(128, 256, 3)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv5_dropout = nn.Dropout(0.4)
        self.conv6 = Conv2dSame(256, 256, 3)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_dropout = nn.Dropout(0.4)
        self.conv7 = Conv2dSame(256, 256, 3)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv8 = Conv2dSame(256, 512, 3)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv8_dropout = nn.Dropout(0.4)
        self.conv9 = Conv2dSame(512, 512, 3)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv9_dropout = nn.Dropout(0.4)
        self.conv10 = Conv2dSame(512, 512, 3)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv11 = Conv2dSame(512, 512, 3)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv11_dropout = nn.Dropout(0.4)
        self.conv12 = Conv2dSame(512, 512, 3)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.conv12_dropout = nn.Dropout(0.4)
        self.conv13 = Conv2dSame(512, 512, 3)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2)

        self.end_dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)
        out = self.conv1_dropout(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = self.maxpool1(out)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = self.conv3_dropout(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)
        out = self.conv5_dropout(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)
        out = self.conv6_dropout(out)
        out = F.relu(self.conv7(out))
        out = self.conv7_bn(out)
        out = self.maxpool3(out)

        out = F.relu(self.conv8(out))
        out = self.conv8_bn(out)
        out = self.conv8_dropout(out)
        out = F.relu(self.conv9(out))
        out = self.conv9_bn(out)
        out = self.conv9_dropout(out)
        out = F.relu(self.conv10(out))
        out = self.conv10_bn(out)
        out = self.maxpool4(out)

        out = F.relu(self.conv11(out))
        out = self.conv11_bn(out)
        out = self.conv11_dropout(out)
        out = F.relu(self.conv12(out))
        out = self.conv12_bn(out)
        out = self.conv12_dropout(out)
        out = F.relu(self.conv13(out))
        out = self.conv13_bn(out)
        out = self.maxpool5(out)

        out = self.end_dropout(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout_fc(out)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    net = VGG_16(10)
    print(net)