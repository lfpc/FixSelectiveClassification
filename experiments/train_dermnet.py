import os
MAIN_PATH = r'/home/luis-felipe'
DATA_PATH = os.path.join(MAIN_PATH,'data')
PATH_MODELS = os.path.join(MAIN_PATH,'torch_models')

import torch
import torchvision
import numpy as np
import wandb
import sys



import models
sys.path.append(os.path.join(MAIN_PATH,'FixSelectiveClassification'))
from utils import measures,metrics
from data_utils import split, accumulate_results
import post_hoc
from models.CIFAR import MEAN,STD
from tqdm import tqdm,trange


#torch.set_default_dtype(torch.float32)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)



# Define o computador utilizado como cuda (gpu) se existir ou cpu caso contrário
print('cuda:', torch.cuda.is_available())
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ["WANDB_SILENT"] = "True"
wandb.login()


CREATE_DIR = True #If true, creates directories to save model (weights_path)
LIVE_PLOT = False #If True, plot* loss while training. If 'print', print loss per epoch
SAVE_CHECKPOINT = True #If True, save (and update) model weights for the best epoch (smallest validation loss)
SAVE_ALL = False #If True, saves weights and trainer at the end of training

DATA = 'DermNet'
PROJECT = 'FixSelectiveClassification'


PRE_TRAINED = True
FREEZE = False
GROUP = DATA
    
VAL_SIZE = 0
BATCH_SIZE = 64

def get_train_transforms(transforms):
    return torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(transforms.resize_size, antialias=True),
    torchvision.transforms.RandomCrop(transforms.crop_size),
    torchvision.transforms.Normalize(transforms.mean,transforms.std),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(30)])


LABEL_SMOOTHING = 0.0
loss_criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
N_EPOCHS = 200

def gain(logits,labels,metric = metrics.N_AURC):
    risk = measures.wrong_class(logits,labels).float()
    p = post_hoc.optimize.p(logits,risk,metric)
    return metric(risk,measures.MSP(logits))-metric(risk,post_hoc.MaxLogit_p(logits,p))
LR = 1e-2

train_data = torchvision.datasets.ImageFolder(os.path.join(DATA_PATH,DATA,'train'), transform=None)
test_data = torchvision.datasets.ImageFolder(os.path.join(DATA_PATH,DATA,'test'), transform=None)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4)
num_classes = len(train_data.classes)
print('N classes = ', num_classes)

risk_dict = {'accuracy': metrics.accuracy,
             'loss': torch.nn.CrossEntropyLoss(),
                'AURC': metrics.AURC_fromlogits,
                'AUROC': metrics.AUROC_fromlogits,
                'ECE': metrics.ECE(15,softmax = True),
                'NormL1': lambda x,y: x.norm(dim=-1,p=1).mean(),
                'Logits Mean': lambda x,y: x.mean(),
                'NormL2': lambda x,y: x.norm(dim=-1,p=2).mean(),
                'NormL4': lambda x,y: x.norm(dim=-1,p=4).mean(),
                'AUROC_gain': gain}


def epoch_fn(model,optimizer,data,loss_criterion):
    '''Train a Neural Network'''
    dev = next(model.parameters()).device
    model.train()
    for image,label in data:
        image,label = image.to(dev), label.to(dev)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_criterion(output,label)
        loss.backward()
        optimizer.step()

def test(model,split,risk_dict:dict) -> dict:
    with torch.no_grad():
        risks = {}
        model.eval()
        if split == 'train':
            logits,labels = accumulate_results(model,train_dataloader)
        elif split =='test' or split == 'validation':
            logits,labels = accumulate_results(model,testloader)
        for name,fn in risk_dict.items():
            risks[split+' '+name] = torch.as_tensor(fn(logits,labels)).item()
    return risks
def save_checkpoint(model,path,name:str):
    name = name+'.pt'
    torch.save(model.state_dict(), os.path.join(path,name))
    print()
    print('saved')
def fine_tune_model(model, freeze:bool,num_classes):
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



def train(MODEL_ARC:str, path_models:str):
    model,transforms_test = models.get_model_imagenet(MODEL_ARC,PRE_TRAINED,True)
    global testloader,train_dataloader
    testloader.dataset.transform = transforms_test
    train_dataloader.dataset.transform = get_train_transforms(transforms_test)

    model = fine_tune_model(model,FREEZE,num_classes).to(dev)

    name = f'{MODEL_ARC}_{DATA}'
    CONFIG = {'Architecture': MODEL_ARC, 'Dataset': DATA, 'N_epochs':N_EPOCHS, 'Validation' : VAL_SIZE, 'LR': LR,
              'Fine-Tunned': PRE_TRAINED, 'Freezed_features': FREEZE}
    WandB = {'project': PROJECT, 'group': GROUP, 'config': CONFIG, 'name': name}

    #optimizer = torch.optim.SGD(list(model.children())[-1].parameters(), lr=LR,momentum = 0.9,weight_decay = 5e-4,nesterov = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay = 2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
    CONFIG['optimizer'] = type(optimizer).__name__
    CONFIG['scheduler'] = type(scheduler).__name__
    wb = wandb.init(reinit = True,**WandB)
    with wb:
        risks = test(model,'train',risk_dict)
        risks.update(test(model,'validation',risk_dict))
        wb.log(risks)
        
        max_acc = risks['validation accuracy']
        pbar = trange(N_EPOCHS,position=0,leave=True, desc = 'Progress:') 
        for e in pbar:
            desc = 'Progress:'
            desc = f"Acc_train: {risks['train accuracy']:.2f} |" +desc   #Loss: {risks['train loss']:.4f} | 
            desc = f"Acc_val (max): {risks['validation accuracy']:.2f} ({max_acc:.2f}) | " + desc
            pbar.set_description(desc)
            #progress_epoch = tqdm(train_dataloader,position=0, leave=False, desc = 'Epoch progress:')
            #progress_epoch.disable = False
            #progress_epoch.reset()
            epoch_fn(model,optimizer,train_dataloader,loss_criterion)
            risks = test(model,'train',risk_dict)
            risks.update(test(model,'validation',risk_dict))
            wb.log(risks)
            #pbar.update()
            scheduler.step()
            if risks['validation accuracy'] >= max_acc:
                save_checkpoint(model,path_models,name)
                max_acc = risks['validation accuracy']

if __name__ == '__main__':
    for MODEL_ARC in [ 'convnext_base','convnext_small','efficientnet_b0',
                      'efficientnet_b1','efficientnet_b3', 'resnet50',
                       'resnet34', 'resnet18', 'wide_resnet50_2',
                      'vgg19_bn', 'vgg16_bn']:
        print(MODEL_ARC)
        path_models = os.path.join(PATH_MODELS,DATA, MODEL_ARC)
        if not os.path.isdir(path_models):
            os.makedirs(path_models)
        train(MODEL_ARC,path_models)
        #else: print('ja foi'); continue