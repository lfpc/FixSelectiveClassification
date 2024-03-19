import os
MAIN_PATH = r'/home/luis-felipe'
DATA_PATH = os.path.join(MAIN_PATH,'data')
PATH_MODELS = os.path.join(MAIN_PATH,'torch_models')

import torch
from torchvision import transforms, datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
import sys



import models
sys.path.append(os.path.join(MAIN_PATH,'FixSelectiveClassification'))
from utils import measures,metrics
from data_utils import split_data, accumulate_results
import post_hoc
from models.CIFAR import MEAN,STD
from tqdm import tqdm,trange

PROJECT = 'FixSelectiveClassification'
GROUP = 'Cifar100'
torch.set_default_dtype(torch.float64)
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

DATA = 'Cifar100'

VAL_SIZE = 0.1
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_data,val_data = split_data.split_dataset(datasets.CIFAR100(
    root=DATA_PATH, train=True, download=True, transform=transform_train),VAL_SIZE,SEED)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
testset = datasets.CIFAR100(
    root=DATA_PATH, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)
num_classes = 100


LABEL_SMOOTHING = 0.0
loss_criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
N_EPOCHS = 200

def gain(logits,labels,metric = metrics.N_AURC):
    risk = measures.wrong_class(logits,labels).float()
    p = post_hoc.optimize.p(logits,risk,metric)
    return metric(risk,measures.MSP(logits))-metric(risk,post_hoc.MaxLogit_p(logits,p))


risk_dict = {'accuracy': metrics.accuracy,
             'loss': loss_criterion,
                'AURC': metrics.AURC_fromlogits,
                'AUROC': metrics.AUROC_fromlogits,
                'ECE': metrics.ECE_0(15,softmax = True),
                'Mean': lambda x,y: x.mean(),
                'NormL2': lambda x,y: x.norm(dim=-1,p=2).mean(),
                'AUROC_gain': gain
                }


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
    risks = {}
    model.eval()
    if split == 'train': data = train_dataloader
    elif split == 'validation': data = testloader
    logits,labels = accumulate_results(model,data)
    for name,fn in risk_dict.items():
        risks[split+' '+name] = torch.as_tensor(fn(logits,labels)).item()
    return risks
def save_checkpoint(model,path,name:str):
    name = name+'.pt'
    torch.save(model.state_dict(), os.path.join(path,name))



def train(MODEL_ARC:str):
    model = models.get_model_cifar(MODEL_ARC,False,False).to(dev)
    path_models = os.path.join(PATH_MODELS,'Cifar100', MODEL_ARC)

    name = f'{MODEL_ARC}_{DATA}'
    CONFIG = {'Architecture': MODEL_ARC, 'Dataset': DATA, 'N_epochs':N_EPOCHS, 'Validation' : VAL_SIZE, 'LabelSmoothing': LABEL_SMOOTHING}
    WandB = {'project': PROJECT, 'group': GROUP, 'config': CONFIG, 'name': name}

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum = 0.9,weight_decay = 5e-4,nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)#StepLR(optimizer, 25, gamma=0.5)
    CONFIG['optimizer'] = type(optimizer).__name__
    CONFIG['scheduler'] = type(scheduler).__name__
    wb = wandb.init(reinit = True,**WandB)
    with wb:
        risks = test(model,'train',risk_dict)
        risks.update(test(model,'validation',risk_dict))
        wb.log(risks)
        progress_epoch = trange(N_EPOCHS,position=0, leave=True, desc = 'Progress:')
        progress = tqdm(train_dataloader,position=1, leave=True, desc = 'Epoch progress:')
        progress_epoch.disable = False
        max_acc = risks['validation accuracy']
        for e in progress_epoch:
            desc = 'Progress:'
            desc = f"Loss: {risks['train loss']:.4f} | Acc_train: {risks['train accuracy']:.2f} |" +desc
            desc = f"Acc_val (max): {risks['validation accuracy']:.2f} ({max_acc:.2f}) | " + desc
            progress_epoch.set_description(desc)
            progress.disable = False
            progress.reset()
            epoch_fn(model,optimizer,progress,loss_criterion)
            risks = test(model,'train',risk_dict)
            risks.update(test(model,'validation',risk_dict))
            wb.log(risks)
            progress_epoch.update()
            scheduler.step()
            if risks['validation accuracy'] >= max_acc:
                save_checkpoint(model,path_models,MODEL_ARC+'_Cifar100')
                max_acc = risks['validation accuracy']

if __name__ == '__main__':
    for MODEL_ARC in models.list_models_cifar():
        print(MODEL_ARC)
        train(MODEL_ARC)