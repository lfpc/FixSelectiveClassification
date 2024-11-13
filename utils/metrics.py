from sklearn.metrics import roc_curve,auc
import torch
from .measures import wrong_class,correct_class,MSP
from numpy import r_,quantile

def accuracy(y_pred,y_true):
    '''Returns the accuracy in a batch'''
    return correct_class(y_pred,y_true).sum()/y_true.size(0)

def ROC_curve(confidence:torch.tensor,risk:torch.tensor, return_threholds = False):
    fpr, tpr, thresholds = roc_curve(risk.cpu(),(1-confidence).cpu())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr
    
def RC_curve(confidence:torch.tensor,risk:torch.tensor, 
             coverages = None, return_thresholds:bool = False):
    risk = risk.view(-1)
    confidence = confidence.view(-1)
    n = len(risk)
    assert len(confidence) == n
    confidence,indices = confidence.sort(descending = True)
    risk = risk[indices]

    if coverages is not None:
        #deprecated
        coverages = torch.as_tensor(coverages,device = risk.device)
        thresholds = confidence.quantile(coverages)
        indices = torch.searchsorted(confidence,thresholds).minimum(torch.as_tensor(confidence.size(0)-1,device=risk.device))
    else:
        #indices = confidence.diff().nonzero().view(-1)
        indices = torch.arange(n,device=risk.device)
    coverages = (1 + indices)/n
    risks = (risk.cumsum(0)[indices])/n
    risks /= coverages
    coverages = r_[0.,coverages.cpu().numpy()]
    risks = r_[0.,risks.cpu().numpy()]

    if return_thresholds:
        thresholds = quantile(confidence.cpu().numpy(),1-coverages)
        return coverages, risks, thresholds
    else: return coverages, risks

def coverages_from_t(g:torch.tensor,t):
    return g.le(t.view(-1,1)).sum(-1)/g.size(0)


def SAC(confidence:torch.tensor,risk:torch.tensor,accuracy):
    coverages,risk = RC_curve(confidence,risk)
    risk = 1-risk #accuracy
    coverages = coverages[risk>=accuracy]
    if coverages.size>0: return coverages[-1]
    else: return 0.0

def AUROC(confidence:torch.tensor,risk:torch.tensor):
    fpr,tpr = ROC_curve(confidence,risk)
    return auc(fpr, tpr)

def AURC(confidence:torch.tensor,risk:torch.tensor, coverages = None):
    coverages,risk_list = RC_curve(confidence,risk, coverages)
    return auc(coverages,risk_list)
def E_AURC(confidence:torch.tensor,risk:torch.tensor, coverages = None):
    return AURC(confidence,risk,coverages)-AURC(1-risk,risk,coverages)

def N_AURC(confidence:torch.tensor,risk:torch.tensor, coverages = None):
    return E_AURC(confidence,risk,coverages)/(risk.mean().item()-AURC(1-risk,risk,coverages))

def AUROC_fromlogits(y_pred,y_true,confidence = None, risk_fn = wrong_class):
    if confidence is None: confidence = MSP(y_pred)
    risk = risk_fn(y_pred,y_true).float()
    return AUROC(confidence,risk)

def AURC_fromlogits(y_pred,y_true,confidence = None, risk_fn = wrong_class, coverages = None):
    if confidence is None: confidence = MSP(y_pred)
    risk = risk_fn(y_pred,y_true).float()
    return AURC(confidence,risk,coverages)

class ECE(torch.nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    def forward(self, confidences:torch.tensor,risk:torch.tensor):
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