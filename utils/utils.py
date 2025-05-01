'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0)
    return mask

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
    return mask

def target_sum(logits_student, logits_teacher, targets, beta=0.5):
    gt_mask = _get_gt_mask(logits_student, targets)
    other_mask = _get_other_mask(logits_student, targets)
    return logits_teacher * beta + gt_mask * logits_student * (1 - beta) + other_mask * logits_teacher * (1 - beta)

def non_target_sum(logits_student, logits_teacher, targets, beta=0.5):
    gt_mask = _get_gt_mask(logits_student, targets)
    other_mask = _get_other_mask(logits_student, targets)
    return logits_teacher * beta + other_mask * logits_student * (1 - beta) + gt_mask * logits_teacher * (1 - beta)
    
def scale_logits(logit_t1, logit_t2):
    pos_logit_t1 = torch.sum(torch.abs(logit_t1), dim=1, keepdim=True) / 2
    pos_logit_t2 = torch.sum(torch.abs(logit_t2), dim=1, keepdim=True) / 2
    return logit_t1 * pos_logit_t2 / pos_logit_t1

def normalize_teacher_logit(z_T, z_V, epsilon=1e-6):
    mu_T, sigma_T = z_T.mean(dim=-1, keepdim=True), z_T.std(dim=-1, keepdim=True)
    mu_V, sigma_V = z_V.mean(dim=-1, keepdim=True), z_V.std(dim=-1, keepdim=True)
    z_T_normalized = mu_V + sigma_V * (z_T - mu_T) / (sigma_T + epsilon)
    return z_T_normalized

def new_weighted_projection(logit_t1, logit_t2, beta=0.5):
    norm_t1 = torch.sqrt(torch.sum(logit_t1 * logit_t1, dim=1, keepdim=True))
    norm_t2 = torch.sqrt(torch.sum(logit_t2 * logit_t2, dim=1, keepdim=True))

    cosine = torch.sum(logit_t1 * logit_t2, dim=1, keepdim=True) / (norm_t1 * norm_t2 + 1e-8)
    coeff = cosine * norm_t1 / norm_t2
    proj_t1_on_t2 = coeff * logit_t2
    
    new_logit = beta * logit_t2 + (1 - beta) * proj_t1_on_t2 # beta * logit_t1 + (1 - beta) * cosine * logit_t2 * norm_t1 / norm_t2
    return new_logit

def weighted_projection(logit_t1, logit_t2, beta=0.5):
    norm_t1 = torch.sqrt(torch.sum(logit_t1 * logit_t1, dim=1, keepdim=True))
    norm_t2 = torch.sqrt(torch.sum(logit_t2 * logit_t2, dim=1, keepdim=True))

    cosine = torch.sum(logit_t1 * logit_t2, dim=1, keepdim=True) / (norm_t1 * norm_t2 + 1e-8)
    coeff = cosine * norm_t1 / norm_t2
    proj_t1_on_t2 = coeff * logit_t2
    
    residual = logit_t1 - proj_t1_on_t2
    new_logit = proj_t1_on_t2 + beta * residual # beta * logit_t1 + (1 - beta) * cosine * logit_t2 * norm_t1 / norm_t2
    return new_logit

def cosine_adaptive(logit_t1, logit_t2, beta=0.5):
    norm_t1 = torch.sqrt(torch.sum(logit_t1 * logit_t1, dim=1, keepdim=True))
    norm_t2 = torch.sqrt(torch.sum(logit_t2 * logit_t2, dim=1, keepdim=True))

    cosine = torch.sum(logit_t1 * logit_t2, dim=1, keepdim=True) / (norm_t1 * norm_t2 + 1e-8)
    
    new_logit = logit_t1 * cosine + (1 - cosine) * logit_t2 
    return new_logit

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def get_cosine_similarity(grads1, grads2, layer_names):
    def cosine_similarity(grad1, grad2):
        dot_product = torch.sum(grad1 * grad2)
        norm_grad1 = torch.norm(grad1)
        norm_grad2 = torch.norm(grad2)
        similarity = dot_product / (norm_grad1 * norm_grad2 + 1e-18) 
        return similarity.item()
    
    cosine_score_with_bias = []
    cosine_score_without_bias = []
    for grad1, grad2, layer_name in zip(grads1, grads2, layer_names):
        if 'bias' not in layer_name:
            cosine_score_without_bias.append(cosine_similarity(grad1, grad2))
        cosine_score_with_bias.append(cosine_similarity(grad1, grad2))
    return np.mean(cosine_score_with_bias), np.mean(cosine_score_without_bias)

def get_grad_norm(grads):
    norm = torch.norm(
                torch.stack([
                    grad.norm(p=2)
                    for grad in grads
                    if grad is not None
                ]),
                p=2
        )
    return norm
    
def get_norm(optimizer):
    logging_dict = {}
    if hasattr(optimizer, 'first_grad_norm'):
        logging_dict['first_grad_norm'] = optimizer.first_grad_norm
    if hasattr(optimizer, 'second_grad_norm'):
        logging_dict['second_grad_norm'] = optimizer.second_grad_norm
    if hasattr(optimizer, 'weight_norm'):
        logging_dict['weight_norm'] = optimizer.weight_norm
    return logging_dict

def get_logging_name(cfg):
    logging_name = ''
    
    logging_name += 'T_MOD'
    for key, value in cfg['teacher_model'].items():
        if isinstance(value, dict):
            for in_key, in_value in value.items():
                if isinstance(in_value, str):
                    _in_value = in_value[:5]
                else: _in_value = in_value
                logging_name += f'_{in_key[:2]}={_in_value}'
        else:
            logging_name += f'_{key[:2]}={value}'
    if 'student_model' in cfg:
        logging_name += 'S_MOD'
        for key, value in cfg['student_model'].items():
            if isinstance(value, dict):
                for in_key, in_value in value.items():
                    if isinstance(in_value, str):
                        _in_value = in_value[:5]
                    else: _in_value = in_value
                    logging_name += f'_{in_key[:2]}={_in_value}'
            else:
                logging_name += f'_{key[:2]}={value}'
        
    logging_name += '_OPT'
    for key, value in cfg['optimizer'].items():
        if isinstance(value, dict):
            for in_key, in_value in value.items():
                if isinstance(in_value, str):
                    _in_value = in_value[:5]
                else: _in_value = in_value
                logging_name += f'_{in_key[:2]}={_in_value}'
        else:
            logging_name += f'_{key[:2]}={value}'
        
    logging_name += '_DAT'
    for key, value in cfg['dataloader'].items():
        if isinstance(value, dict):
            for in_key, in_value in value.items():
                if isinstance(in_value, str):
                    _in_value = in_value[:5]
                else: _in_value = in_value
                logging_name += f'_{in_key[:2]}={_in_value}'
        else:
            logging_name += f'_{key[:2]}={value}'
        
    return logging_name

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    np.random.seed(seed)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 80  # default terminal width
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
