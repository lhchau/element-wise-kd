import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, T: int = 2, soft_target_loss_weight: float = 0.25, ce_loss_weight: float = 0.75):
        """
        Label Smoothing Loss

        Args:
            num_classes (int): Number of classes.
            smoothing (float): Smoothing factor, typically between 0.0 and 1.0.
        """
        super(KDLoss, self).__init__()
        assert T >= 1, "Temperature T should be in [1, +infty)"
        self.T = T
        self.ce_criterion = nn.CrossEntropyLoss()
        self.soft_target_loss_weight = soft_target_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.ce_loss = 0
        self.kl_loss = 0

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, mode='normal') -> torch.Tensor:
        if mode == 'normal':
            soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=-1)
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T**2)
            
            label_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = (self.soft_target_loss_weight * soft_targets_loss).item()
            self.ce_loss = (self.ce_loss_weight * label_loss).item()
            return self.soft_target_loss_weight * soft_targets_loss + self.ce_loss_weight * label_loss
        
        elif mode == 'ce':
            return self.ce_loss_weight * self.ce_criterion(student_logits, target)
        
        elif mode == 'kd':
            soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=-1)
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T**2)
            return self.soft_target_loss_weight * soft_targets_loss
