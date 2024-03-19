from sklearn.metrics import roc_curve,auc
import torch
from .measures import wrong_class,correct_class,MSP
from numpy import r_,quantile

def accuracy(y_pred,y_true):
    '''Returns the accuracy in a batch'''
    return correct_class(y_pred,y_true).sum()/y_true.size(0)


def ROC_curve(loss, confidence, return_threholds = False):
    fpr, tpr, thresholds = roc_curve(loss.cpu(),(1-confidence).cpu())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr
    
def RC_curve(loss:torch.tensor, confidence:torch.tensor,
             coverages = None, return_thresholds:bool = False):
    loss = loss.view(-1)
    confidence = confidence.view(-1)
    n = len(loss)
    assert len(confidence) == n
    confidence,indices = confidence.sort(descending = True)
    loss = loss[indices]

    if coverages is not None:
        #deprecated
        coverages = torch.as_tensor(coverages,device = loss.device)
        thresholds = confidence.quantile(coverages)
        indices = torch.searchsorted(confidence,thresholds).minimum(torch.as_tensor(confidence.size(0)-1,device=loss.device))
    else:
        #indices = confidence.diff().nonzero().view(-1)
        indices = torch.arange(n,device=loss.device)
    coverages = (1 + indices)/n
    risks = (loss.cumsum(0)[indices])/n
    risks /= coverages
    coverages = r_[0.,coverages.cpu().numpy()]
    risks = r_[0.,risks.cpu().numpy()]

    if return_thresholds:
        thresholds = quantile(confidence.cpu().numpy(),1-coverages)
        return coverages, risks, thresholds
    else: return coverages, risks

def coverages_from_t(g:torch.tensor,t):
    return g.le(t.view(-1,1)).sum(-1)/g.size(0)


def SAC(risk:torch.tensor,confidence:torch.tensor,accuracy):
    coverages,risk = RC_curve(risk,confidence)
    risk = 1-risk #accuracy
    coverages = coverages[risk>=accuracy]
    if coverages.size>0: return coverages[-1]
    else: return 0.0

def AUROC(loss,confidence):
    fpr,tpr = ROC_curve(loss,confidence)
    return auc(fpr, tpr)

def AURC(loss,confidence, coverages = None):
    coverages,risk_list = RC_curve(loss,confidence, coverages)
    return auc(coverages,risk_list)
def E_AURC(loss,confidence, coverages = None):
    return AURC(loss,confidence,coverages)-AURC(loss,1-loss,coverages)
def N_AURC(loss,confidence, coverages = None):
    return E_AURC(loss,confidence,coverages)/(loss.mean().item()-AURC(loss,1-loss,coverages))

def AUROC_fromlogits(y_pred,y_true,confidence = None, risk_fn = wrong_class):
    if confidence is None: confidence = MSP(y_pred)
    risk = risk_fn(y_pred,y_true).float()
    return AUROC(risk,confidence)

def AURC_fromlogits(y_pred,y_true,confidence = None, risk_fn = wrong_class, coverages = None):
    if confidence is None: confidence = MSP(y_pred)
    risk = risk_fn(y_pred,y_true).float()
    return AURC(risk,confidence,coverages)



class ECE_0(torch.nn.Module):
    
    '''From https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py :'''
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence y_preds into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10, softmax = True):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.SM = softmax

    def forward(self, y:torch.tensor, labels):
        if self.SM:
            y = y.softmax(-1)
        confidences, predictions = torch.max(y, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=y.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.item()

class ECE(torch.nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    def forward(self, confidences:torch.tensor, risk):
        risk = 1-risk

        ece = torch.zeros(1, device=risk.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = risk[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.item()
    def diagram(self,confidences:torch.tensor,risk:torch.tensor):
        risk = 1-risk

        confs = []
        accs = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accs.append(risk[in_bin].float().mean().item())
                confs.append(confidences[in_bin].mean().item())
            
        return confs,accs

class AdaptiveECE(torch.nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECE, self).__init__()
        self.n_bins = n_bins

    def get_bounds(self, x):
        return x.quantile(torch.linspace(0,1,self.n_bins+1,device=x.device))
    def from_logits(self,logits,labels):
        return self.forward(MSP(logits),wrong_class(logits,labels).float())
    def get_indices(self,x:torch.tensor):
        return torch.linspace(0,x.size(0),self.n_bins+1,device=x.device).long()
    def get_bins(self,confidence:torch.tensor,errors:torch.tensor):
        confidence,indices = confidence.sort()
        errors = errors[indices]
        confs = []
        accs = []
        indices = self.get_indices(confidence)
        for i,b1 in enumerate(indices[:-1]):
            b2 = indices[i+1]
            confs.append(confidence[b1:b2].mean())
            accs.append(1-(errors[b1:b2].sum()/(b2-b1)))
        return torch.tensor(confs),torch.tensor(accs)
    def forward(self, confidence:torch.tensor,errors:torch.tensor):
        confs,accs = self.get_bins(confidence,errors)
        return (confs-accs).abs().mean()

        




class ClasswiseECE(torch.nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15, softmax = True):
        super(ClasswiseECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.SM = softmax

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        if self.SM:
            logits = torch.nn.functional.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = logits[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce