import torch
import os
from .utils import *
from .bypass_bn import *


def loop_one_epoch_knowledge_distillation_samkd(dataloader, student, teacher, criterion, optimizer, device, logging_dict, epoch):
    loss, total, correct, correct5 = 0, 0, 0, 0
    ce_loss, kl_loss = 0, 0
    teacher.eval()
    student.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        outputs = student(inputs)
        first_loss = criterion(teacher_logits, outputs, targets, 'gap')
        optimizer.zero_grad()
        first_loss.backward(retain_graph=True)
        optimizer.first_step(zero_grad=True)
        criterion(teacher_logits, student(inputs), targets, 'normal').backward()
        optimizer.second_step()
        ce_loss += criterion.ce_loss
        kl_loss += criterion.kl_loss
        if (batch_idx + 1) == len(dataloader):
            logging_dict['Train/ce_loss'] = ce_loss / (batch_idx + 1)
            logging_dict['Train/kl_loss'] = kl_loss / (batch_idx + 1)
        with torch.no_grad():
            loss += first_loss.item()
            loss_mean = loss / (batch_idx + 1)
            
            correct_top1, correct_top5 = accuracy(outputs, targets, topk=(1, 5))
            correct += correct_top1
            correct5 += correct_top5
            total += targets.size(0)
            acc = 100. * correct / total
            acc5 = 100. * correct5 / total
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
    logging_dict[f'Train/loss'] = loss_mean
    logging_dict[f'Train/acc5'] = acc5
    logging_dict[f'Train/acc'] = acc