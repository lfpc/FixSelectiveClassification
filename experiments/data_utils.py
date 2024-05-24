import torch
from os.path import join,exists
from os import listdir
from models import get_model
from torchvision import datasets
from torch.utils.data import Subset, Dataset,random_split, DataLoader

MODELS_DIR = join(r'/home/luis-felipe','torch_models')

def accumulate_results(model,data, set_eval = False):
    '''Accumulate output (of model) and label of a entire dataset.'''

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if set_eval:
        model.eval()

    output_list = []
    label_list = []
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev,dtype), label.to(dev)

            label_list.append(label)
            output_list.append(model(image))
    output_list = torch.cat(output_list)
    label_list = torch.cat(label_list)
        
    return output_list,label_list

def get_dataloader(data:str, split = 'test', batch_size = 100, data_dir = r'/data', transforms = None,subset = False, 
                   num_workers:int = 4):
    if data.lower() == 'imagenet':
        if exists(join(data_dir,'ImageNet')): data_dir = join(data_dir,'ImageNet')
        if 'corrupted' in split:
            if isinstance(split, tuple): split = join(split[0],split[1],split[2])
        elif split == 'v2': split = 'imagenetv2-matched-frequency'
        elif split == 'test': split = 'val'
        dataset = datasets.imagenet.ImageFolder(join(data_dir,split),transform=transforms)
        #dataset = datasets.imagenet.ImageNet(data_dir,split = split,transform = transforms)
        
    elif data.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir,
                                    train=(split=='train'),
                                    download=True,
                                    transform=transforms)
    elif data.lower() == 'oxfordiiitpet':
        dataset = datasets.OxfordIIITPet(root=join(data_dir),
                                    split = 'test' if split=='test' else 'trainval',
                                    download=True,
                                    transform=transforms)
    elif data.lower() == 'dermnet':
        dataset = datasets.ImageFolder(join(data_dir,data,split), transform=transforms)
    if subset:
        dataset = split_data(subset, len(dataset)).dataset(dataset)[1]
    return DataLoader(dataset,batch_size=batch_size, num_workers=num_workers)


def calc_logits(model_arc:str,data:str = 'ImageNet',models_dir= MODELS_DIR, 
                  split = 'val', device = torch.device('cuda'), **kwargs_data):
    classifier,transforms = get_model(model_arc,data,True,True,join(models_dir))
    classifier = classifier.to(device).eval()
    dataloader = get_dataloader(data,split,transforms = transforms,**kwargs_data)
    logits,labels =  accumulate_results(classifier,dataloader)
    models_dir = join(models_dir,data)
    if isinstance(split,tuple): 
        models_dir = join(models_dir,split[0])
        split = '_'.join(split)
    else: models_dir = join(models_dir,split)

    torch.save(logits, join(models_dir,'logits',f'{model_arc}_{data}_{split}_logits.pt'))
    torch.save(labels,join(models_dir,'labels',f'{data}_{split}_labels.pt'))
    return logits.to(torch.get_default_dtype()),labels

def upload_logits(model_arc:str,data:str = 'ImageNet',models_dir= MODELS_DIR, 
                  split = 'val', device = torch.device('cuda'), 
                  **kwargs_data):
    if split == 'test': split = 'val'
    if isinstance(split,tuple): 
        split_str = '_'.join(split)
        split_folder = split[0]
    else: 
        split_str = split
        split_folder = split
    if f'{model_arc}_{data}_{split_str}_logits.pt' in listdir(join(models_dir,data,split_folder,'logits')):
        logits = torch.load(join(models_dir,data,split_folder,'logits',f'{model_arc}_{data}_{split_str}_logits.pt')).to(device)
        labels = torch.load(join(models_dir,data,split_folder,'labels',f'{data}_{split_str}_labels.pt')).to(device)
        return logits.to(torch.get_default_dtype()),labels
    else: return calc_logits(model_arc, data, models_dir,split, device,**kwargs_data)
        

class split_data():
    def __init__(self,validation_size, n = 50000, seed = 42):
        if validation_size<1:
            validation_size = validation_size*n
        assert validation_size <= n
        self.val_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[:int(validation_size)]
        self.test_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[int(validation_size):]
    def logits(self,logits,labels):
        logits_val,labels_val = logits[self.val_index],labels[self.val_index]
        logits_test,labels_test = logits[self.test_index],labels[self.test_index]
        return logits_val,labels_val,logits_test,labels_test 
    def dataset(self,dataset):
        return Subset(dataset,self.test_index), Subset(dataset,self.val_index)
    @staticmethod
    def split_logits(logits,labels,validation_size = 0.1,seed:int = 42):
        n = labels.size(0)
        val_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[:int(validation_size*n)]
        test_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[int(validation_size*n):]
        logits_val,labels_val = logits[val_index],labels[val_index]
        logits_test,labels_test = logits[test_index],labels[test_index]
        return logits_val,labels_val,logits_test,labels_test
    @staticmethod
    def split_dataset(dataset,validation_size:float = 0.1, seed:int = 42):
        return random_split(dataset, [1-validation_size, validation_size], generator = torch.Generator().manual_seed(seed))
    @staticmethod
    def split_logits_balanced(logits:torch.tensor,labels:torch.tensor,validation_size:float, seed = 42):
        unique,  num = torch.unique(labels,return_counts=True)
        labels_val = []
        logits_val = []
        labels_test = []
        logits_test = []
        for i,l in enumerate(unique):
            n = int(num[i]*validation_size)
            index_val = torch.randperm(num[i],generator = torch.Generator().manual_seed(seed))[:n]
            index_test = torch.randperm(num[i],generator = torch.Generator().manual_seed(seed))[n:]
            labels_i = labels[labels==l.item()]
            logits_i = logits[labels==l.item()]
            labels_val.append(labels_i[index_val])
            labels_test.append(labels_i[index_test])
            logits_val.append(logits_i[index_val])
            logits_test.append(logits_i[index_test])
        return torch.cat(logits_val),torch.cat(labels_val),torch.cat(logits_test),torch.cat(labels_test)
    
