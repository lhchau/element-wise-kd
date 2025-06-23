import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_entropy(p_t):
    entropy = - (p_t * (p_t + 1e-12).log())
    return entropy

class KD(nn.Module):
    def __init__(self, T: int = 2, kl_weight: float = 0.25, ce_weight: float = 0.75, mode='normal'):
        super(KD, self).__init__()
        assert T >= 1, "Temperature T should be in [1, +infty)"
        self.T = T
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.ce_loss = 0
        self.kl_loss = 0
        self.mode = mode

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, target: torch.Tensor, epoch) -> torch.Tensor:
        if self.mode == 'normal':
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            ce_loss = self.ce_criterion(student_logits, target)
            
            self.kl_loss = kl_loss.item()
            self.ce_loss = ce_loss.item()
            return self.kl_weight * kl_loss + self.ce_weight * ce_loss
        
        elif self.mode == 'ce':
            ce_loss = self.ce_criterion(student_logits, target)
            self.ce_loss = ce_loss.item()
            return self.ce_weight * ce_loss
        
        elif self.mode == 'kl':
            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            
            self.kl_loss = kl_loss.item()
            return self.kl_weight * kl_loss

        elif self.mode == 'ablation':
            self.ce_criterion = nn.CrossEntropyLoss(reduction='none')

            p_s = F.log_softmax(student_logits / self.T, dim=-1)
            p_t = F.softmax(teacher_logits / self.T, dim=-1)
            
            entropy = compute_entropy(p_t).sum(1)
            percent = 0.05
            num_samples = entropy.size(0)
            k = max(1, int(num_samples * percent))
            _, lowest_entropy_indices = torch.topk(entropy, k, largest=False)
            
            mask = torch.ones(num_samples, dtype=torch.bool, device=entropy.device)
            mask[lowest_entropy_indices] = False
            flipped_mask = ~mask
            
            kl_loss = F.kl_div(p_s, p_t, reduction='none').sum(1) * (self.T**2)
            ce_loss = self.ce_criterion(student_logits, target)
            
            final_loss = self.kl_weight * kl_loss * mask + self.ce_weight * ce_loss * mask + ce_loss * flipped_mask 
            final_loss = final_loss.mean()
            
            self.kl_loss = kl_loss.mean().item()
            self.ce_loss = ce_loss.mean().item()
            return final_loss
        
        elif self.mode == 'ablation_add':
            ADD_CONST = 2
            with torch.no_grad():
                p_t = F.softmax(teacher_logits / self.T, dim=-1)

                entropy = compute_entropy(p_t).sum(1)
                percent = 0.05
                num_samples = entropy.size(0)
                k = max(1, int(num_samples * percent))
                _, lowest_entropy_indices = torch.topk(entropy, k, largest=False)
                
                mask = torch.ones(num_samples, dtype=torch.bool, device=entropy.device)
                mask[lowest_entropy_indices] = False
                flipped_mask = ~mask
            
            new_student_logits = student_logits * mask.unsqueeze(-1) / self.T + student_logits * flipped_mask.unsqueeze(-1) / (self.T + ADD_CONST)
            p_s = F.log_softmax(new_student_logits, dim=-1)
            
            new_teacher_logits = teacher_logits * mask.unsqueeze(-1) / self.T + teacher_logits * flipped_mask.unsqueeze(-1) / (self.T + ADD_CONST)
            p_t = F.softmax(new_teacher_logits, dim=-1)
            
            kl_loss = F.kl_div(p_s, p_t, reduction='none').sum(1)
            kl_loss = kl_loss * mask * (self.T**2) + kl_loss * flipped_mask * ((self.T+ADD_CONST)**2)
            ce_loss = self.ce_criterion(student_logits, target)
            
            final_loss = self.kl_weight * kl_loss.mean() + self.ce_weight * ce_loss
            
            self.kl_loss = kl_loss.mean().item()
            self.ce_loss = ce_loss.item()
            return final_loss