import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scale_normalize(logit):
    with torch.no_grad():
        mean = logit.mean(dim=-1, keepdims=True)
    return (logit - mean)

class ATS(nn.Module):
    def __init__(self, kl_weight: float = 0.25, ce_weight: float = 0.75, rho=20, mode='normal'):
        super(ATS, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.ce_loss = 0
        self.kl_loss = 0
        self.rho = rho
        self.mode = mode

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, epoch) -> torch.Tensor:
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
        
        log_p_s = F.log_softmax(student_logits / stu_temp.unsqueeze(1), dim=-1)
        p_t = F.softmax(teacher_logits / tea_temp.unsqueeze(1), dim=-1)
        kl_loss = F.kl_div(log_p_s, p_t, reduction='none') * (stu_temp * tea_temp).unsqueeze(1) / teacher_logits.shape[0]
        kl_loss = kl_loss.sum()
        
        self.kl_loss = kl_loss.item()
        self.ce_loss = ce_loss.item()
        return self.kl_weight * kl_loss + self.ce_weight * ce_loss