
import torch
from torch.nn import functional as F

y_true =torch.tensor([[0.0, 1.0], [0.0, 1.0]])
y_pred =torch.tensor([[8.0, 6.0], [3.0, 9.0]])

def binary_ce(a_v, p_v, logit):
    if logit:
        p_v = p_v.exp() / (1 + p_v.exp())
    pv1=(F.softmax(p_v, 1)).log()
    pv2=F.log_softmax(p_v, 1)
    return -(a_v * p_v.log() + (1-a_v) * (1 - p_v).log())

def binary_crossentropy(A, P, logit=False):
    #return torch.mean([binary_ce(a_i, p_i, logit) for a_i, p_i in zip(A, P)])
    loss=binary_ce(A, P, logit)
    return torch.mean(loss)

loss_b=binary_crossentropy(y_true,y_pred,logit=False)
loss_b1=F.binary_cross_entropy(y_pred,y_true.float())
print("over")