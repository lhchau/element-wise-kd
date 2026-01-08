import torch
import os
from .utils import *
from .bypass_bn import *


def loop_one_epoch_knowledge_distillation(dataloader, student, teacher, criterion, optimizer, device, logging_dict, epoch):
    loss, total, correct, correct5 = 0, 0, 0, 0
    ce_loss, kl_loss = 0, 0
    teacher.eval()
    student.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        outputs = student(inputs)
        first_loss = criterion(teacher_logits, outputs, targets, epoch)
        first_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
            if batch_idx % 3 == 0 or (batch_idx + 1) == len(dataloader):
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
        # del teacher_logits, outputs, inputs, targets
        # torch.cuda.empty_cache()
        
    logging_dict[f'Train/loss'] = loss_mean
    logging_dict[f'Train/acc5'] = acc5
    logging_dict[f'Train/acc'] = acc
    

def loop_one_epoch(dataloader, model, criterion, optimizer, device, logging_dict, epoch, loop_type='train', logging_name=None, best_acc=0):
    loss, total, correct, correct5 = 0, 0, 0, 0
    if loop_type == 'train': 
        model.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            first_loss = criterion(outputs, targets)
            first_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                loss += first_loss.item()
                loss_mean = loss / (batch_idx + 1)
                
                correct_top1, correct_top5 = accuracy(outputs, targets, topk=(1, 5))
                correct += correct_top1
                correct5 += correct_top5
                total += targets.size(0)
                acc = 100. * correct / total
                acc5 = 100. * correct5 / total
                if batch_idx % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
        # del outputs, inputs, targets
        # torch.cuda.empty_cache()
                
    elif loop_type == 'test':
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                first_loss = criterion(outputs, targets)

                loss += first_loss.item()
                loss_mean = loss / (batch_idx + 1)
                
                correct_top1, correct_top5 = accuracy(outputs, targets, topk=(1, 5))
                correct += correct_top1
                correct5 += correct_top5
                total += targets.size(0)
                acc = 100. * correct / total
                acc5 = 100. * correct5 / total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'loss': loss_mean,
                'epoch': epoch
            }
            save_path = os.path.join('checkpoint', logging_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            if acc > best_acc:
                best_acc = acc
                print('Saving best checkpoint ...')
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
            if epoch == 240:
                print(f'Saving checkpoint at epoch {epoch} ...')
                torch.save(state, os.path.join(save_path, f'ckpt_{epoch}.pth'))
        logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
        logging_dict[f'{loop_type.title()}/gen_gap'] = logging_dict['Train/acc'] - acc
        
    else: 
        raise ValueError(f'Do not have implementation for loop type: {loop_type}')
    
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc5'] = acc5
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'test': 
        return best_acc, acc