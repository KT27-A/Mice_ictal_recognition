import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    # def __init__(self, alpha=None, gamma=0, size_average=True, criterion=None):
    #     super(FocalLoss, self).__init__()
    #     self.alpha = alpha
    #     self.gamma = gamma
        
    #     if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
    #     if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
    #     self.size_average = size_average
    #     self.criterion = criterion

    def __init__(self,
                 alpha=0.75,
                 gamma=2,
                 criterion=None,
                 reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criterion = criterion
        self.cal_p = nn.Softmax(dim=1)

    # def forward(self, input, target):
    #     # if input.dim()>2:
    #     #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
    #     #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
    #     #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    #     # target = target.view(-1,1)
    #     outputs = self.criterion(input, target)

    #     logpt = F.log_softmax(input, dim=1)
    #     import pdb; pdb.set_trace()
    #     # logpt = logpt.gather(1,target)
    #     # logpt = logpt.view(-1)
    #     pt = Variable(logpt.data.exp())

    #     # if self.alpha is not None:
    #     #     if self.alpha.type()!=input.data.type():
    #     #         self.alpha = self.alpha.type_as(input.data)
    #     #     at = self.alpha.gather(0,target.data.view(-1))
    #     #     logpt = logpt * Variable(at)

    #     loss = -1 * (1-pt)**self.gamma * logpt
    #     if self.size_average: return loss.mean()
    #     else: return loss.sum()

    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(self.alpha[0])
            alpha[label == 1] = self.alpha[1]

        # probs = torch.sigmoid(logits, dim=1)
        probs = self.cal_p(logits)
        
        pt = torch.where(label == 1, probs[:, 1], probs[:, 0])
        ce_loss = self.criterion(logits, label)
        loss = (alpha[:, 1] * torch.pow(1 - pt, self.gamma) * ce_loss)
        # loss = (alpha[:, 1] * ce_loss)
        # loss = (torch.pow(1 - pt, self.gamma) * ce_loss)

        # alpha
        # loss = (alpha[:, 1] * ce_loss)
        
        # loss = ce_loss

        # 0, (1-p)
        # zeros = torch.zeros(probs[:, 1].shape).cuda()
        # pt = torch.where(label == 1, zeros, 1-probs[:, 1])
        # loss = (alpha[:, 1] * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss