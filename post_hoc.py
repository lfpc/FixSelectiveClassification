import torch
from utils.metrics import AURC
import numpy as np
from utils.measures import MSP,max_logit,entropy

def centralize(y:torch.tensor):
    return y-(y.mean(-1).view(-1,1))

def p_norm(y:torch.tensor,p, eps:float = 1e-25):
    if p is None or p == 0: return torch.ones(y.size(0),1,device=y.device)
    else: return y.norm(p=p,dim=-1).clamp_min(eps).view(-1,1)
 
def normalize(logits:torch.tensor,p):
    if p is None or p==0: return logits
    else: return torch.nn.functional.normalize(logits,p,-1)

def temperature_scaling(logits:torch.tensor,T:float):
    assert T > 0
    return logits.div(T)

def p_TS(logits:torch.tensor,p,T):
    return normalize(logits,p).div(T)

def MaxLogit_p(logits:torch.tensor,
               p = 'optimal',
               msp_fallback = False,
               return_p = False,
               centralize_logits = True,
               **kwargs_optimize):
    if centralize_logits: logits = centralize(logits)
    logits_opt = kwargs_optimize.pop('logits_opt',logits)
    if p == 'optimal':
        p = optimize.p(logits_opt,method = max_logit,**kwargs_optimize)
    ML_p = max_logit(normalize(logits,p))
    if msp_fallback:
        assert 'metric' in kwargs_optimize and 'risk' in kwargs_optimize
        fallback = (kwargs_optimize['metric'](kwargs_optimize['risk'],max_logit(normalize(logits_opt,p))) > kwargs_optimize['metric'](kwargs_optimize['risk'],MSP(logits_opt)))
        if fallback:
            ML_p =  MSP(logits)
            p = None
    if return_p: return ML_p,p
    else: return ML_p

def MSP_p(logits:torch.tensor,
               pt = 'optimal',
               return_pt = False,
               centralize_logits = True,
               **kwargs_optimize):
    if centralize_logits: logits = centralize(logits)
    if pt == 'optimal':
        pt = optimize.p_and_beta(logits,method = MSP,**kwargs_optimize)
    msp_p = MSP(normalize(logits,pt[0]).div(pt[1]))
    if return_pt: return msp_p,pt
    else: return msp_p

def significant(x,epsilon:float = 0.01):
    return x*(x > epsilon)


class optimize:
    '''Gradient methods could be used, but a grid search
    on a small set of p's show to be strongly efficient for pNorm optimization.
    Also, AURC and AUROC are not differentiable'''
    p_range = torch.arange(9)
    T_range = torch.arange(0.01,2,0.01)
    @staticmethod
    def p_and_beta(logits,risk,method = MSP,metric = AURC,
                   p_range = p_range,T_range =T_range):
        T_range = torch.as_tensor(T_range)
        vals = optimize.p_T_grid(logits,risk,method,metric,p_range,T_range)
        p,T = np.unravel_index(np.argmin(vals),np.shape(vals))
        p = p_range[p]
        T = T_range.div(p_norm(logits,p).mean().cpu())[T]
        
        return p,T
    @staticmethod
    def p(logits, risk,method = max_logit,metric = AURC,p_range = p_range):
        vals = optimize.p_grid(logits,risk,method,metric,p_range)
        p = p_range[np.argmin(vals)]
        return p
    @staticmethod
    def T(logits, risk,method = MSP,metric = AURC,T_range = T_range):
        vals = optimize.T_grid(logits,risk,method,metric,T_range)
        return T_range[np.argmin(vals)]
    @staticmethod
    def T_grid(logits,risk,method = MSP,metric = AURC,T_range = T_range):
        vals = []
        for T in T_range:
            vals.append(metric(risk,method(logits.div(T))).item())
        return vals
    @staticmethod
    def p_grid(logits,risk,method = MSP,metric = AURC,p_range = p_range):
        vals = []
        for p in p_range:
            vals.append(metric(risk,method(normalize(logits,p))).item())
        return vals
    @staticmethod
    def p_T_grid(logits,risk,method = MSP,metric = AURC,p_range = p_range,T_range = T_range):
        vals = []
        for p in p_range:
            vals_T = optimize.T_grid(normalize(logits,p),risk,method,metric,T_range.div(p_norm(logits,p).mean().cpu()))
            vals.append(vals_T)
        return vals
    @staticmethod
    def T_fromloss(logits,labels,metric = torch.nn.CrossEntropyLoss(),T_range = T_range):
        vals = optimize.T_grid_fromloss(logits,labels,metric,T_range)
        return T_range[np.argmin(vals)]
    @staticmethod
    def T_grid_fromloss(logits,labels,metric = torch.nn.CrossEntropyLoss(),T_range = T_range):
        vals = []
        for T in T_range:
            vals.append(metric(logits.div(T),labels).item())
        return vals
    
class other_methods():
    @staticmethod
    def LDA(logits:torch.tensor, eps = 1e-20):
        logits = logits.softmax(-1)
        p1 = logits.max(-1).values.reshape(-1,1)
        sigma2 = (logits.size(-1)**2)*logits[logits!=logits.max(-1).values.reshape(-1,1)].reshape(-1,logits.size(-1)-1).var() 
        return (p1 - logits).sum(-1).pow(2) / (sigma2 + eps)
    @staticmethod
    def BK(logits:torch.tensor, alpha, beta):
        p = logits.softmax(-1).topk(2,dim=-1).values.T
        return alpha*p[0]-beta*p[1]
    @staticmethod
    def J(logits:torch.tensor):
        logits = logits.softmax(-1)
        j1 = logits.max(-1).values
        logits = logits-j1.reshape(-1,1)
        return j1 - logits.mean(-1) - logits.std(-1) 
    @staticmethod
    def ETS(logits:torch.tensor,T,w1,w2):
        return w1*logits.div(T).softmax(-1).max(-1).values + w2*logits.softmax(-1).max(-1).values
    @staticmethod
    def HTS(logits:torch.tensor,w,b):
        normalized_entropy = entropy(logits).div(np.log(logits.size(-1)))
        T_h = (normalized_entropy.log()*w+b.exp()+1).log().reshape(-1,1)
        return MSP(logits.div(T_h))
class optimize_other_methods():
    @staticmethod
    def ETS(logits,risk,T = None,w_range = torch.arange(0,1.01,0.01),metric = AURC, **kwargs_T):
        vals = []
        if T is None:
            T = optimize.T(logits, risk,**kwargs_T)
        for w1 in w_range:
            vals.append(metric(risk,other_methods.ETS(logits,T,w1,1-w1)).item())
        w1 = w_range[np.argmin(vals)]
        return T,w1,1-w1
    @staticmethod
    def BK(logits:torch.tensor,risk,alpha_range = torch.arange(0,1,0.1),beta_range = torch.arange(-1,1,0.01), metric = AURC):
        vals = []
        for alpha in alpha_range:
            vals_b = []
            for beta in beta_range:
                vals_b.append(metric(risk,other_methods.BK(logits,alpha,beta)).item())
            vals.append(vals_b)
            alpha,beta = np.unravel_index(np.argmin(vals),np.shape(vals))
        return alpha_range[alpha],beta_range[beta]
    @staticmethod
    def HTS(logits:torch.tensor,risk,w_range = torch.arange(-1,1,0.1),b_range = torch.arange(-3,1,0.01), metric = AURC):
        vals = []
        for w in w_range:
            vals_b = []
            for b in b_range:
                vals_b.append(metric(risk,other_methods.HTS(logits,w=w,b=b)).item())
            vals.append(vals_b)
        w,b = np.unravel_index(np.argmin(vals),np.shape(vals))
        return w_range[w],b_range[b]