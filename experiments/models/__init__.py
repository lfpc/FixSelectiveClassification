from .CIFAR import list_models as list_models_cifar
from .CIFAR import get_model as get_model_cifar
from .ImageNet import list_models as list_models_imagenet
from .ImageNet import get_model as get_model_imagenet
from .OxfordPets import get_model as get_model_oxfordpets

def list_models(data:str = 'ImageNet'):
    if data.lower() == 'imagenet':
        return list_models_imagenet()
    elif data.lower() == 'cifar100':
        return list_models_cifar()
    
def get_model(MODEL_ARC:str,data:str = 'ImageNet',pretrained = True,
              return_transforms = True,weights_path = None):
    
    if data.lower() == 'imagenet':
        return get_model_imagenet(MODEL_ARC,pretrained,return_transforms)
    elif data.lower() == 'cifar100':
        return get_model_cifar(MODEL_ARC,pretrained,return_transforms, weights_path)
    elif data.lower() == 'oxfordiiitpet':
        return get_model_oxfordpets(MODEL_ARC,weights_path,pretrained,return_transforms)