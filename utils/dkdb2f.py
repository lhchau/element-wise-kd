import torch
import torch.nn as nn
import torch.nn.functional as F

def get_cosine(z1, z2):
    norm_z1 = torch.norm(z1, p=2, dim=1, keepdim=True)
    norm_z2 = torch.norm(z2, p=2, dim=1, keepdim=True)
    cosine = torch.sum(z1 * z2, dim=1, keepdim=True) / (norm_z1 * norm_z2 + 1e-8)
    return cosine

class DKDB2F(nn.Module):
    def __init__(self, T: int = 2, kl_weight: float = 0.25, ce_weight: float = 0.75, alpha: float = 0.5, dkd_alpha: float = 1, dkd_beta: float = 1):
        super(DKDB2F, self).__init__()
        assert T >= 1, "Temperature T should be in [1, +infty)"
        self.T = T
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.dkd_alpha = dkd_alpha
        self.dkd_beta = dkd_beta
        self.tckd_loss = 0
        self.nckd_loss = 0
        self.ce_loss = 0
        self.kl_loss = 0
        self.new_kl_loss = 0

    def forward(self, teacher_logits: torch.Tensor, self_teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, mode='normal') -> torch.Tensor:
        if mode == 'normal':        
            ce_loss = self.ce_criterion(student_logits, target)
            
            new_teacher_logits = self.alpha * teacher_logits + (1 - self.alpha) * self_teacher_logits
            with torch.no_grad():
                p_t = F.softmax(teacher_logits / self.T, dim=-1)
                p_s = F.log_softmax(student_logits / self.T, dim=-1)
                new_p_t = F.softmax(new_teacher_logits / self.T, dim=-1)
                new_kl_loss = F.kl_div(p_s, new_p_t, reduction='batchmean') * (self.T**2)
                kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            nckd_loss = compute_nckd(student_logits, new_teacher_logits, target, self.T)
            tckd_loss = compute_tckd(student_logits, new_teacher_logits, target, self.T)
            self.ce_loss = ce_loss.item()
            self.kl_loss = kl_loss.item()
            self.new_kl_loss = new_kl_loss.item()
            self.nckd_loss = nckd_loss.item()
            self.tckd_loss = tckd_loss.item()
            return self.ce_weight * ce_loss + self.dkd_alpha * tckd_loss + self.dkd_beta * nckd_loss
        
def compute_nckd(logits_student, logits_teacher, target, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)
    return nckd_loss

def compute_tckd(logits_student, logits_teacher, target, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (temperature**2) 
    return tckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
