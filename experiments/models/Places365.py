import torchvision.models as torch_models
import torchvision.transforms as torch_transforms
import os
import torch

def_path = os.path.join(r'/home','luis-felipe','torch_models','Places365')
transforms = torch_transforms.Compose([
        torch_transforms.Resize((256,256)),
        torch_transforms.CenterCrop(224),
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_weight(MODEL_ARC:str, path=def_path):
    model_file = '%s_places365.pth.tar' % MODEL_ARC
    if not os.access(os.path.join(path,model_file), os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)
    checkpoint = torch.load(os.path.join(path,model_file), map_location=lambda storage, loc: storage)
    return {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

def get_model(MODEL_ARC:str,path=def_path, pretrained:bool = True, return_transforms:bool = True):
    if pretrained:
        weights = get_weight(MODEL_ARC,path)
    else: 
        weights = None
    model = torch_models.__dict__[MODEL_ARC](num_classes = 365)
    model.load_state_dict(weights)
    model.eval()
    
    if return_transforms: return model,transforms
    else: return model