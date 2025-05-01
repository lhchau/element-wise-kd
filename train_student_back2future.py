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
# Config
cfg = exec_configurator()
if cfg['student_model']['model_name'] in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    cfg['optimizer']['lr'] = 0.01
# Initialization
initialize(cfg['trainer']['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, logging_dict = 0, {}
EPOCHS = cfg['trainer']['epochs'] 
# Model path
teacher_path = cfg['trainer'].get('teacher_path', None)
self_teacher_path = cfg['trainer'].get('self_teacher_path', None)
assert teacher_path is not None, "Teacher path should not be None"
assert self_teacher_path is not None, 'Self-Teacher path should not be None'
# KL Hyperparameters
T = cfg['trainer'].get('T', 4)
kl_weight = cfg['trainer'].get('kl_weight', 0.9)
ce_weight = cfg['trainer'].get('ce_weight', 1)
alpha = cfg['trainer'].get('alpha', None)
# Logging
logging_name = get_logging_name(cfg)
logging_name += (f'_Temp={T}' + f'_kl_w={kl_weight}' + f'_ce_w={ce_weight}' + f'_b2f_al={alpha}' + f'_{current_time}')
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
#### 2.2 Self-Teacher
################################
self_teacher_model_name = cfg['trainer'].get('self_teacher', None)
if self_teacher_model_name == None:
    self_teacher_model_name = cfg['student_model']['model_name']
self_teacher_model = model_dict[self_teacher_model_name](num_classes=num_classes)
self_teacher_model = self_teacher_model.to(device)
# Self-Teacher - Load checkpoint
print(f'==> Load checkpoint for Self-Teacher: {self_teacher_model_name}')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(self_teacher_path)
self_teacher_model.load_state_dict(checkpoint['model'])

################################
#### 2.3 Student
################################
student_model_name = cfg['student_model'].pop('model_name', None)
student_model = model_dict[student_model_name](num_classes=num_classes, **cfg['student_model'])
student_model = student_model.to(device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = B2F(T=T, kl_weight=kl_weight, ce_weight=ce_weight, alpha=alpha)
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
        loop_one_epoch_back2future(
            dataloader=train_dataloader,
            student=student_model,
            teacher=teacher_model,
            self_teacher=self_teacher_model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            mode='order_teacher')
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