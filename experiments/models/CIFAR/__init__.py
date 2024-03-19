'''Addapted from https://github.com/kuangliu/pytorch-cifar'''

from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *
from .wide_resnet import *
from torchvision import transforms as torch_transforms
from os.path import join
from os import listdir

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def list_models():
    return ['DenseNet121', 'DenseNet169','DenseNet201', 'DenseNet161', 'SimpleDLA', 
            'DLA', 'DPN26','DPN92', 'EfficientNetB0', 'GoogLeNet','LeNet',
        'MobileNet','MobileNetV2','PNASNetA','PNASNetB','PreActResNet18', 'PreActResNet34',
        'PreActResNet50','PreActResNet101','PreActResNet152',
        'RegNetX_200MF','RegNetX_400MF','RegNetY_400MF','ResNet18','ResNet34','ResNet50',
        'ResNet101','ResNet152','ResNeXt29_2x64d','ResNeXt29_4x64d',
        'ResNeXt29_8x64d','ResNeXt29_32x4d','SENet18','ShuffleNetG2','ShuffleNetG3',
        'ShuffleNetV2','VGG_11','VGG_13','VGG_16','VGG_19','WideResNet28_10']

def get_model(MODEL_ARC:str, pretrained:bool = True, 
              return_transforms:bool = True, weights_path = ''):
    
    model = globals()[MODEL_ARC](num_classes=100)
    transforms =  torch_transforms.Compose([
    torch_transforms.ToTensor(),
    torch_transforms.Normalize(MEAN, STD),])

    if pretrained:
        weights_path = join(weights_path,MODEL_ARC)
        f = listdir(weights_path)[0]
        weights_path = join(weights_path,f)
        model.load_state_dict(torch.load(weights_path))
    if return_transforms:   return model,transforms
    else: return model