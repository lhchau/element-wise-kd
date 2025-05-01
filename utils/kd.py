import torch
import torch.nn as nn
import torch.nn.functional as F

class KD(nn.Module):
    def __init__(self, T: int = 2, kl_weight: float = 0.25, ce_weight: float = 0.75):
        super(KD, self).__init__()
        assert T >= 1, "Temperature T should be in [1, +infty)"
        self.T = T
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.ce_loss = 0
        self.kl_loss = 0

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, mode='normal') -> torch.Tensor:
        if mode == 'normal':
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * kl_loss + self.ce_weight * ce_loss
        
        elif mode == 'ce':
            ce_loss = self.ce_criterion(student_logits, target)
            self.ce_loss = ce_loss.item()
            return self.ce_weight * ce_loss
        
        elif mode == 'kl':
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            self.kl_loss = kl_loss.item()
            return self.kl_weight * kl_loss

        elif mode == 'gap':
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            # self.gap_loss = kl_loss.item()
            # self.ce_loss = ce_loss.item()
            return self.kl_weight * kl_loss - self.ce_weight * ce_loss