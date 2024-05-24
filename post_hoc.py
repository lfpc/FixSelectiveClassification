import torch
from utils.metrics import AURC
from utils.measures import MSP,max_logit

def centralize(logits:torch.tensor):
    return logits-(logits.mean(-1).view(-1,1))

def p_norm(logits:torch.tensor,p, eps:float = 1e-12):
    return logits.norm(p=p,dim=-1).clamp_min(eps).view(-1,1)
 
def normalize(logits:torch.tensor,p, centralize_logits:bool = True):
    assert not torch.any(torch.any(logits,-1).logical_not()) #assert logits are not all zeros
    if centralize_logits: 
        logits = centralize(logits)
    return torch.nn.functional.normalize(logits,p,-1)

def temperature_scaling(logits:torch.tensor,T:float):
    assert T > 0
    return logits.div(T)

def p_TS(logits:torch.tensor,p,T):
    return normalize(logits,p).div(T)

def MaxLogit_pNorm(logits:torch.tensor,
               p = 'optimal',
               centralize_logits:bool = True,
               **kwargs_optimize):
    if centralize_logits: logits = centralize(logits)
    if p == 'optimal':
        p = optimize.p(kwargs_optimize.pop('logits_opt',logits),kwargs_optimize.pop('risk'),centralize_logits=False,**kwargs_optimize)
    if p == 'MSP': return MSP(logits)
    else: return max_logit(normalize(logits,p,False))

def significant(x,epsilon:float = 0.01):
    return x*(x > epsilon)


class optimize:
    p_range = torch.arange(10)
    T_range = torch.arange(0.01,2,0.01)
        
    @staticmethod
    def T(logits:torch.tensor, risk:torch.tensor,method = MSP,metric = AURC,T_range = T_range):
        metric_min = torch.inf
        t_opt = 1
        for t in T_range:
            metric_value = metric(method(logits.div(t)),risk)
            if metric_value < metric_min:
                metric_min = metric_value
                t_opt = t
        return t_opt
    
    @staticmethod
    def p(logits:torch.tensor, risk:torch.tensor,method = max_logit,metric = AURC,p_range = p_range,
           MSP_fallback:bool = True, centralize_logits:bool = True):
        metric_min = metric(MSP(logits),risk) if MSP_fallback else torch.inf
        p_opt = 'MSP' if MSP_fallback else 0
        for p in p_range:
            metric_value = metric(method(normalize(logits,p,centralize_logits)),risk)
            if metric_value < metric_min:
                metric_min = metric_value
                p_opt = p
        return p_opt

    @staticmethod
    def p_and_T(logits:torch.tensor, risk:torch.tensor,method = MSP,metric = AURC,p_range = p_range,T_range = T_range,
                 centralize_logits:bool = True, rescale_T:bool = True):
        metric_min = torch.inf
        t_opt = 1
        p_opt = None
        if centralize_logits: logits = centralize(logits)
        for p in p_range:
            norm = p_norm(logits,p)
            for t in T_range:
                if rescale_T: t = t / norm.mean()
                metric_value = metric(method(logits.div(t*norm)),risk)
                if metric_value < metric_min:
                    metric_min = metric_value
                    t_opt = t
                    p_opt = p
        return p_opt,t_opt