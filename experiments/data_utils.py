import torch
from os.path import join,exists
from os import listdir
from models import get_model
from torchvision import datasets
from torch.utils.data import Subset, Dataset,random_split, DataLoader

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

def get_dataloader(DATA:str, split = 'test', batch_size = 100, data_dir = r'/data', transforms = None,subset = False, 
                   num_workers:int = 4):
    if DATA.lower() == 'imagenet':
        if exists(join(data_dir,'ImageNet')): data_dir = join(data_dir,'ImageNet')
        if 'corrupted' in split:
            if isinstance(split, tuple): split = join(split[0],split[1],split[2])
        elif split == 'v2': split = 'imagenetv2-matched-frequency'
        elif split == 'test': split = 'val'
        dataset = datasets.imagenet.ImageFolder(join(data_dir,split),transform=transforms)
        #dataset = datasets.imagenet.ImageNet(data_dir,split = split,transform = transforms)
        
    elif DATA.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir,
                                    train=(split=='train'),
                                    download=True,
                                    transform=transforms)
    elif DATA.lower() == 'places365':
        dataset = datasets.Places365(root=join(data_dir,'Places365'),
                                    split = split,
                                    small=True,
                                    transform=transforms)
    elif DATA.lower() == 'oxfordiiitpet':
        dataset = datasets.OxfordIIITPet(root=join(data_dir),
                                    split = 'test' if split=='test' else 'trainval',
                                    download=True,
                                    transform=transforms)
    elif DATA.lower() == 'dermnet':
        dataset = datasets.ImageFolder(join(data_dir,DATA,split), transform=transforms)
    if subset:
        dataset = split_data(subset, len(dataset)).dataset(dataset)[1]
    return DataLoader(dataset,batch_size=batch_size, num_workers=num_workers)


def calc_logits(model_arc:str,DATA:str = 'ImageNet',models_dir= r'/models', 
                  split = 'test', device = torch.device('cuda'), **kwargs_data):
    classifier,transforms = get_model(model_arc,DATA,True,True,join(models_dir))
    classifier = classifier.to(device).eval()
    dataloader = get_dataloader(DATA,split,transforms = transforms,**kwargs_data)
    outputs,labels =  accumulate_results(classifier,dataloader)
    if isinstance(split,tuple): split = '_'.join(split)
    torch.save(outputs, join(models_dir,'outputs',f'{model_arc}_{DATA}_outputs_{split}.pt'))
    torch.save(labels,join(models_dir,'outputs',f'{DATA}_labels_{split}.pt'))
    return outputs.to(torch.get_default_dtype()),labels

def upload_logits(model_arc:str,DATA:str = 'ImageNet',models_dir= r'/models', 
                  split = 'test', device = torch.device('cuda'), **kwargs_data):
    models_dir = join(models_dir,DATA)
    if isinstance(split,tuple): split_str = '_'.join(split)
    else: split_str = split
    if f'{model_arc}_{DATA}_outputs_{split_str}.pt' in listdir(join(models_dir,'outputs')):
        outputs = torch.load(join(models_dir,'outputs',f'{model_arc}_{DATA}_outputs_{split_str}.pt')).to(device)
        labels = torch.load(join(models_dir,'outputs',f'{DATA}_labels_{split_str}.pt')).to(device)
        return outputs.to(torch.get_default_dtype()),labels
    else: return calc_logits(model_arc, DATA, models_dir,split, device,**kwargs_data)
        

class split_data():
    def __init__(self,validation_size, n = 50000, seed = 42):
        if validation_size<1:
            validation_size = validation_size*n
        assert validation_size <= n
        self.val_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[:int(validation_size)]
        self.test_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[int(validation_size):]
    def logits(self,outputs,labels):
        outputs_val,labels_val = outputs[self.val_index],labels[self.val_index]
        outputs_test,labels_test = outputs[self.test_index],labels[self.test_index]
        return outputs_val,labels_val,outputs_test,labels_test 
    def dataset(self,dataset):
        return Subset(dataset,self.test_index), Subset(dataset,self.val_index)
    @staticmethod
    def split_logits(outputs,labels,validation_size = 0.1,seed:int = 42):
        n = labels.size(0)
        val_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[:int(validation_size*n)]
        test_index = torch.randperm(n,generator = torch.Generator().manual_seed(seed))[int(validation_size*n):]
        outputs_val,labels_val = outputs[val_index],labels[val_index]
        outputs_test,labels_test = outputs[test_index],labels[test_index]
        return outputs_val,labels_val,outputs_test,labels_test
    @staticmethod
    def split_dataset(dataset,validation_size:float = 0.1, seed:int = 42):
        return random_split(dataset, [1-validation_size, validation_size], generator = torch.Generator().manual_seed(seed))
    
