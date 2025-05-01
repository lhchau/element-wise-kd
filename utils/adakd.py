import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ADAKD(nn.Module):
    def __init__(self, kl_weight: float = 0.25, ce_weight: float = 0.75, rho=20):
        super(ADAKD, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.ce_loss = 0
        self.kl_loss = 0
        self.rho = rho

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, mode='normal') -> torch.Tensor:
        if mode == 'normal':
            max_logit, _ = teacher_logits.max(dim=1)
            T = max_logit / math.log(self.rho)
            
            p_s = F.log_softmax(student_logits / T.unsqueeze(1), dim=-1)
            p_t = F.softmax(teacher_logits / T.unsqueeze(1), dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='none') * (T * T).unsqueeze(1) / teacher_logits.shape[0]
            kl_loss = kl_loss.sum()
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * kl_loss + self.ce_weight * ce_loss