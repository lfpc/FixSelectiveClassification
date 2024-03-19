import torchvision.models as torch_models
import torchvision.transforms as torch_transforms
import os
import torch
from .ImageNet import get_model as get_model_imagenet

def_path = os.path.join(r'/home','luis-felipe','torch_models','OxfordIIITPet')

def fine_tune_model(model, freeze:bool = False, PRE_TRAINED = True):
    name,classifier = list(model.named_children())[-1]
    if freeze and PRE_TRAINED:
        for param in model.parameters():
            param.requires_grad = False
    else:
        model.train()
    if isinstance(classifier,torch.nn.Linear):
        in_f = classifier.in_features
        model._modules[name] = torch.nn.Linear(in_f, 37)
    else:
        in_f = classifier[-1].in_features
        classifier[-1] = torch.nn.Linear(in_f, 37)
        for param in classifier.parameters():
            param.requires_grad = True
    return model

def get_weight(MODEL_ARC:str, path=def_path):
    return torch.load(os.path.join(path,MODEL_ARC,f'{MODEL_ARC}_OxfordIIITPet.pt'))

def get_model(MODEL_ARC:str,path=def_path, pretrained:bool = True, return_transforms:bool = True):
    if pretrained:
        weights = get_weight(MODEL_ARC,path)
    else: 
        weights = None
    model,transforms_test = get_model_imagenet(MODEL_ARC,True,True)
    model = fine_tune_model(model,False,False)
    model.load_state_dict(weights)
    model.eval()
    
    if return_transforms: return model,transforms_test
    else: return model