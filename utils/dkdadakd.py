import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scale_normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    return (logit - mean)

class DKDADAKD(nn.Module):
    def __init__(self, T: int = 2, dkd_alpha: float = 1, dkd_beta: float = 2, ce_weight: float = 1, kl_weight: float = 1, rho=3, mode='normal', warmup=1):
        super(DKDADAKD, self).__init__()
        assert T >= 1, "Temperature T should be in [1, +infty)"
        self.T = T
        self.ce_criterion = nn.CrossEntropyLoss()
        self.dkd_alpha = dkd_alpha
        self.dkd_beta = dkd_beta
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.tckd_loss = 0
        self.nckd_loss = 0
        self.kl_loss = 0
        self.ce_loss = 0
        self.rho = rho
        self.mode = mode
        self.warmup = warmup

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, epoch) -> torch.Tensor:
        if self.mode == 'normal':
            ce_loss = self.ce_criterion(student_logits, target)
            
            teacher_logits = scale_normalize(teacher_logits)
            student_logits = scale_normalize(student_logits)
            
            tea_max_logit, _ = teacher_logits.max(dim=1)
            tea_temp = tea_max_logit / self.rho
            tea_temp.clip_(1, None)
            
            with torch.no_grad():
                stu_max_logit, _ = student_logits.max(dim=1)
                stu_temp = stu_max_logit / self.rho
                stu_temp.clip_(1, None)
                
            nckd_loss = compute_nckd(student_logits, teacher_logits, target, tea_temp, stu_temp)
            tckd_loss = compute_tckd(student_logits, teacher_logits, target, tea_temp, stu_temp)
            self.ce_loss = ce_loss.item()
            self.nckd_loss = nckd_loss.item()
            self.tckd_loss = tckd_loss.item()
            return self.ce_weight * ce_loss + min(epoch / self.warmup, 1) * (self.dkd_alpha * tckd_loss + self.dkd_beta * nckd_loss * self.kl_weight)
        
def compute_nckd(logits_student, logits_teacher, target, tea_temp, stu_temp):
    gt_mask = _get_gt_mask(logits_student, target)
    pred_teacher_part2 = F.softmax(logits_teacher / tea_temp.unsqueeze(1) - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / stu_temp.unsqueeze(1) - 1000.0 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none') * (tea_temp * stu_temp).unsqueeze(1) / logits_teacher.shape[0]
    return nckd_loss.sum()

def compute_tckd(logits_student, logits_teacher, target, tea_temp, stu_temp):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / stu_temp.unsqueeze(1), dim=1)
    pred_teacher = F.softmax(logits_teacher / tea_temp.unsqueeze(1), dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='none') * (stu_temp * tea_temp).unsqueeze(1) / logits_teacher.shape[0]
    return tckd_loss.sum()

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
