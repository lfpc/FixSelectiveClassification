import torch

'''Confidence measures from an output vector. 
Note that some measures return confidence (max, margin),
while others return uncertainty (entropy, energy).
Conversion between both can be made with a signal exchange'''

def MSP(y:torch.tensor):
    return y.softmax(-1).max(-1).values

def entropy(y:torch.tensor):
    return torch.special.entr(y.softmax(-1)).sum(-1)

def negative_entropy(y:torch.tensor):
    return entropy(y).mul(-1)

def energy(z:torch.Tensor, T = 1.0):
    return z.div(T).logsumexp(-1).mul(-T)
    #return -T*((z/T).exp().sum(-1).log())

def max_logit(y:torch.tensor):
    return y.max(-1).values

def margin_logits(y:torch.tensor):
    return torch.sub(*y.topk(2,dim=-1).values.t())

def margin_softmax(y:torch.tensor):
    return margin_logits(y.softmax(-1))

def gini(logits:torch.tensor):
    return 1-logits.softmax(-1).pow(2).sum(-1)
def negative_gini(logits:torch.tensor):
    return logits.softmax(-1).pow(2).sum(-1)-1

def softmax_k(logits:torch.tensor,k:int):
    return logits.topk(k,dim=-1).values.softmax(-1).max(-1).values
def k_logit(logits:torch.tensor,k:int):
    return logits.topk(k,dim=-1).values[:,-1]

def TCP(y_pred:torch.tensor,y_true:torch.tensor):
    ''' Returns the True Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class that is true'''
    return y_pred.index_select(-1,y_true).diag()

    

def predicted_class(y_pred):
    '''Returns the predicted class for a given output.'''
    with torch.no_grad():
        if y_pred.shape[-1] == 1:
            y_pred = y_pred.view(-1)
            y_pred = (y_pred>0.5).int()
            
        else:
            y_pred = torch.argmax(y_pred, -1)
    return y_pred

def correct_class(y_pred,y_true):
    '''Returns a bool tensor indicating if each prediction is correct'''
    with torch.no_grad():
        y_pred = predicted_class(y_pred)
        correct = y_pred.eq(y_true)
    
    return correct

def wrong_class(y_pred,y_true):
    '''Returns a bool tensor indicating if each prediction is wrong'''
    return correct_class(y_pred,y_true).logical_not()

