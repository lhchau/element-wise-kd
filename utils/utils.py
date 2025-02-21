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
from PIL import Image
import torchvision.transforms as transforms

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
