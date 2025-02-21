import wandb
import datetime
import pprint

import torch
import torch.nn as nn

from models import *
from utils import *
from dataloaders import *
from optimizers import *

################################
#### 0. SETUP CONFIGURATION
################################
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

cfg = exec_configurator()
initialize(cfg['trainer']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch, logging_dict = 0, 0, {}

# Total number of training epochs
EPOCHS = cfg['trainer']['epochs'] 
teacher_path = cfg['trainer'].get('teacher_path', None)
if teacher_path is None:
    raise ValueError('Teacher path should not be None')

# KL Hyperparameters
T = cfg['trainer'].get('T', 4)
kl_weight = cfg['trainer'].get('kl_weight', 0.9)
ce_weight = cfg['trainer'].get('ce_weight', 1)

# Logging
print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += (f'_Temp={T}' + f'_kl_w={kl_weight}' + f'_ce_w={ce_weight}')
logging_name += ('_' + current_time)

framework_name = cfg['logging']['framework_name']
if framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)
pprint.pprint(cfg)

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])

################################
#### 2. BUILD THE NEURAL NETWORK
################################
# Teacher
teacher_model_name = cfg['teacher_model'].pop('model_name', None)
teacher_model = model_dict[teacher_model_name](num_classes=num_classes, **cfg['teacher_model'])
teacher_model = teacher_model.to(device)
total_params = sum(p.numel() for p in teacher_model.parameters())
print(f'==> Number of parameters in Teacher: {cfg["teacher_model"]}: {total_params}')

# Teacher - Load checkpoint
print('==> Resuming from best checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
load_path = os.path.join('checkpoint', teacher_path)
checkpoint = torch.load(os.path.join(load_path, 'ckpt_best.pth'))
teacher_model.load_state_dict(checkpoint['net'])

# Student
student_model_name = cfg['student_model'].pop('model_name', None)
student_model = model_dict[student_model_name](num_classes=num_classes, **cfg['student_model'])
student_model = student_model.to(device)
total_params = sum(p.numel() for p in student_model.parameters())
print(f'==> Number of parameters in Student: {cfg["student_model"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = KDLoss(T=T, kl_weight=kl_weight, ce_weight=ce_weight)
test_criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(student_model, opt_name, cfg['optimizer'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

################################
#### 3.b Training 
################################
if __name__ == "__main__":
    for epoch in range(start_epoch, EPOCHS):
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
            net=student_model,
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