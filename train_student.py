import wandb
import datetime
import pprint

import torch
from torch import nn, optim

from models import *
from utils import *
from dataloaders import *
from optimizers import *

################################
#### 0. SETUP CONFIGURATION
################################
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Config
cfg = exec_configurator()
if cfg['student_model']['model_name'] in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    cfg['optimizer']['lr'] = 0.01
# Initialization
initialize(cfg['trainer']['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, logging_dict = 0, {}
EPOCHS = cfg['trainer']['epochs'] 
teacher_path = cfg['trainer'].get('teacher_path', None)
assert teacher_path is not None, "Teacher path should not be None"
# KL Hyperparameters

kd_type = cfg['trainer'].get('kd_type', 'kd')
T = cfg['trainer'].get('T', 4)
kl_weight = cfg['trainer'].get('kl_weight', 0.9)
ce_weight = cfg['trainer'].get('ce_weight', 1)
rho = cfg['trainer'].get('rho', None)
# Logging
print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += (f'_Temp={T}' + f'_kl_w={kl_weight}' + f'_ce_w={ce_weight}' + f'_kd_t={kd_type}' + f'_rho={rho}' + f'_{current_time}')
framework_name = cfg['logging']['framework_name']
if framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)
pprint.pprint(cfg)

#### 1. BUILD THE DATASET
################################
train_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])

################################
#### 2. BUILD THE NEURAL NETWORK
################################
#### 2.1 Teacher
################################
teacher_model_name = cfg['teacher_model'].pop('model_name', None)
teacher_model = model_dict[teacher_model_name](num_classes=num_classes, **cfg['teacher_model'])
teacher_model = teacher_model.to(device)
# Teacher - Load checkpoint
print(f'==> Load checkpoint for Teacher: {teacher_model_name}')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(teacher_path)
teacher_model.load_state_dict(checkpoint['model'])

################################
#### 2.2 Student
################################
student_model_name = cfg['student_model'].pop('model_name', None)
student_model = model_dict[student_model_name](num_classes=num_classes, **cfg['student_model'])
student_model = student_model.to(device)
total_params = sum(p.numel() for p in student_model.parameters())
print(f'==> Number of parameters in Student: {student_model_name}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
if kd_type == 'kd':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight)
elif kd_type == 'adakd':
    criterion = ADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho)
elif kd_type == 'dkdadakd':
    criterion = DKDADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho)
elif kd_type == 'adakd_lsd':
    criterion = ADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho, mode='lsd')
elif kd_type == 'adakd_gap':
    criterion = ADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho, mode='gap')
elif kd_type == 'both_temp':
    criterion = ADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho, mode='both_temp')
elif kd_type == 're_both_temp':
    criterion = ADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho, mode='re_both_temp')

test_criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(student_model, opt_name, cfg['optimizer'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

################################
#### 3.b Training 
################################
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        print('\nEpoch: %d' % epoch)
        loop_one_epoch_knowledge_distillation(
            dataloader=train_dataloader,
            student=student_model,
            teacher=teacher_model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch)
        best_acc, acc = loop_one_epoch(
            dataloader=test_dataloader,
            model=student_model,
            criterion=test_criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name,
            best_acc=best_acc)
        if scheduler is not None:
            scheduler.step()
        
        if framework_name == 'wandb':
            wandb.log(logging_dict)