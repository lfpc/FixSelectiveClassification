import torchvision.models as torch_models
import torchvision.transforms as torch_transforms
import timm

timm_special_models ={
    'efficientnetv2_xl': 'tf_efficientnetv2_xl.in21k_ft_in1k',
    'vit_l_16_384':'vit_large_patch16_384.augreg_in21k_ft_in1k',
    'vit_b_16_sam':'vit_base_patch16_224.sam',
    'vit_b_32_sam': 'vit_base_patch32_224.sam'
}

def efficientnetv2_xl(weights = True,**kwargs):
    if hasattr(weights,'pretrained'):
        pretrained = weights.pretrained
    else: pretrained = weights
    return timm.create_model(timm_special_models['efficientnetv2_xl'],pretrained=pretrained,**kwargs)
def vit_l_16_384(weights = True,**kwargs):
    if hasattr(weights,'pretrained'):
        pretrained = weights.pretrained
    else: pretrained = weights
    return timm.create_model(timm_special_models['vit_l_16_384'],pretrained=pretrained,**kwargs)
def vit_b_16_sam(weights = True,**kwargs):
    if hasattr(weights,'pretrained'):
        pretrained = weights.pretrained
    else: pretrained = weights
    return timm.create_model(timm_special_models['vit_b_16_sam'],pretrained=pretrained,**kwargs)
def vit_b_32_sam(weights = True,**kwargs):
    if hasattr(weights,'pretrained'):
        pretrained = weights.pretrained
    else: pretrained = weights
    return timm.create_model(timm_special_models['vit_b_32_sam'],pretrained=pretrained,**kwargs)

class timm_weights():
    def __init__(self, model:str):
        if model in timm_special_models.keys():
            model = timm_special_models[model]
        self.pretrained = timm.is_model_pretrained(model)
        self.model = model
    def transforms(self):
        transform = timm.data.create_transform(**timm.data.resolve_data_config(timm.get_pretrained_cfg(self.model).__dict__))
        return transform
    
def get_weight(model:str,weight:str = 'DEFAULT'):
    if model in torch_models.list_models():
        return torch_models.get_model_weights(model).__dict__[weight]
    elif timm.is_model_pretrained(model) or model in timm_special_models.keys():
        return timm_weights(model)

def list_models():
    models = torch_models.list_models(module=torch_models)
    models.extend(list(timm_special_models.keys()))
    return models

def default_transforms():
    return torch_transforms.Compose([
    torch_transforms.Resize(256),
    torch_transforms.CenterCrop(224),
    torch_transforms.ToTensor(),
    torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

def get_model(MODEL_ARC:str, pretrained:bool = True, return_transforms:bool = True):
    if MODEL_ARC in torch_models.list_models():
        if pretrained:
            weights = get_weight(MODEL_ARC)
            transforms = weights.transforms()
        else: 
            weights = None
            transforms = default_transforms()
        model = torch_models.__dict__[MODEL_ARC](weights = weights)
    elif MODEL_ARC in timm_special_models.keys():
        MODEL_ARC = timm_special_models[MODEL_ARC]
        transforms = timm.data.create_transform(**timm.data.resolve_data_config(timm.get_pretrained_cfg(MODEL_ARC).__dict__))
        model = timm.create_model(MODEL_ARC,pretrained)
    elif MODEL_ARC in timm.list_models():
        transforms = timm.data.create_transform(**timm.data.resolve_data_config(timm.get_pretrained_cfg(MODEL_ARC).__dict__))
        model = timm.create_model(MODEL_ARC,pretrained)
    
    if return_transforms: return model,transforms
    else: return model
    


