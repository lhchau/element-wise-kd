import torch
import torch.nn as nn
import torch.nn.functional as F

def get_cosine(z1, z2):
    norm_z1 = torch.norm(z1, p=2, dim=1, keepdim=True)
    norm_z2 = torch.norm(z2, p=2, dim=1, keepdim=True)
    cosine = torch.sum(z1 * z2, dim=1, keepdim=True) / (norm_z1 * norm_z2 + 1e-8)
    return cosine

class B2F(nn.Module):
    def __init__(self, T: int = 2, kl_weight: float = 0.25, ce_weight: float = 0.75, alpha: float = 0.5):
        super(B2F, self).__init__()
        assert T >= 1, "Temperature T should be in [1, +infty)"
        self.T = T
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.ce_loss = 0
        self.kl_loss = 0
        self.new_kl_loss = 0

    def forward(self, teacher_logits: torch.Tensor, self_teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, mode='normal') -> torch.Tensor:
        if mode == 'normal':        
            new_teacher_logits = self.alpha * teacher_logits + (1 - self.alpha) * self_teacher_logits
        
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            new_p_t = F.softmax(new_teacher_logits / self.T, dim=-1)
            new_kl_loss = F.kl_div(p_s, new_p_t, reduction='batchmean') * (self.T**2)
            with torch.no_grad():
                kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.new_kl_loss = new_kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * new_kl_loss + self.ce_weight * ce_loss
        elif mode == 'ce':
            ce_loss = self.ce_criterion(student_logits, target)
            self.ce_loss = ce_loss.item()
            return self.ce_weight * ce_loss
        elif mode == 'kl':
            new_teacher_logits = self.alpha * teacher_logits + (1 - self.alpha) * self_teacher_logits
        
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            new_p_t = F.softmax(new_teacher_logits / self.T, dim=-1)
            new_kl_loss = F.kl_div(p_s, new_p_t, reduction='batchmean') * (self.T**2)
            with torch.no_grad():
                kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            self.kl_loss = kl_loss.item()
            self.new_kl_loss = new_kl_loss.item()
            return self.kl_weight * new_kl_loss
        
        elif mode == 'only_correct':
            with torch.no_grad():
                predictions = self_teacher_logits.argmax(1)
                correct_mask = predictions == target
        
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='none').sum(1) * (self.T**2) / teacher_logits.shape[0]
            kl_loss = kl_loss * correct_mask
            kl_loss = kl_loss.sum()
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * kl_loss + self.ce_weight * ce_loss
        
        elif mode == 'prob':
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            self_p_t = F.softmax(self_teacher_logits / self.T, dim=-1)
            new_p_t = self.alpha * p_t + (1 - self.alpha) * self_p_t
            new_kl_loss = F.kl_div(p_s, new_p_t, reduction='batchmean') * (self.T**2)
            with torch.no_grad():
                kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.new_kl_loss = new_kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * new_kl_loss + self.ce_weight * ce_loss
        
        elif mode == 'order_teacher':
            values, _ = self_teacher_logits.sort(dim=1)
            _, order = teacher_logits.sort(dim=1)
            new_teacher_logits = torch.zeros_like(teacher_logits)
            new_teacher_logits.scatter_(1, order, values)
            
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            new_p_t = F.softmax(new_teacher_logits / self.T, dim=-1)
            new_kl_loss = F.kl_div(p_s, new_p_t, reduction='batchmean') * (self.T**2)
            with torch.no_grad():
                kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.new_kl_loss = new_kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * new_kl_loss + self.ce_weight * ce_loss
        
        elif mode == 'order_self':
            values, _ = teacher_logits.sort(dim=1)
            _, order = self_teacher_logits.sort(dim=1)
            new_teacher_logits = torch.zeros_like(teacher_logits)
            new_teacher_logits.scatter_(1, order, values)
            
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            new_p_t = F.softmax(new_teacher_logits / self.T, dim=-1)
            new_kl_loss = F.kl_div(p_s, new_p_t, reduction='batchmean') * (self.T**2)
            with torch.no_grad():
                kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.new_kl_loss = new_kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * new_kl_loss + self.ce_weight * ce_loss
        
    def set_alpha(self, alpha):
        self.alpha = alpha