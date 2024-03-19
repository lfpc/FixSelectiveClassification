import torch
from .ImageNet import get_model as get_model_imagenet
from os.path import join

def fine_tune_model(model, freeze:bool, num_classes:int, PRE_TRAINED = True):
    name,classifier = list(model.named_children())[-1]
    if freeze and PRE_TRAINED:
        for param in model.parameters():
            param.requires_grad = False
    else:
        model.train()
    if isinstance(classifier,torch.nn.Linear):
        in_f = classifier.in_features
        model._modules[name] = torch.nn.Linear(in_f, num_classes)
    else:
        in_f = classifier[-1].in_features
        classifier[-1] = torch.nn.Linear(in_f, num_classes)
        for param in classifier.parameters():
            param.requires_grad = True
    return model

def get_model(MODEL_ARC:str,path, pretrained:bool = True, return_transforms:bool = True):
    model,transforms = get_model_imagenet(MODEL_ARC,True,True)
    model = fine_tune_model(model,freeze = False,num_classes = 23)
    if pretrained:
        model.load_state_dict(torch.load(join(path,MODEL_ARC,f'{MODEL_ARC}_DermNet.pt')))
    model.eval()
    if return_transforms: return model,transforms
    else: return model