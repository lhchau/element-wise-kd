import wandb
import datetime
import pprint

import torch
import torch.nn as nn

from models import *
from utils import *
from dataloaders import *
from optimizers import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
cfg = exec_configurator()
initialize(cfg['trainer']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch, logging_dict = 0, 0, {}

# Total number of training epochs
EPOCHS = cfg['trainer']['epochs'] 
scheduler = cfg['trainer'].get('scheduler', None)
teacher_path = cfg['trainer'].get('teacher_path', None)
if teacher_path is None:
    raise ValueError('Teacher path should not be None')
T = cfg['trainer'].get('T', 2)
soft_target_loss_weight = cfg['trainer'].get('soft_target_loss_weight', None)
ce_loss_weight = cfg['trainer'].get('ce_loss_weight', None)

# Logging
print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += (f'_Temp={T}' + f'_soft_w={soft_target_loss_weight}' + f'_ce_w={ce_loss_weight}')
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
teacher_model = get_model(**cfg['teacher_model'], num_classes=num_classes)
teacher_model = teacher_model.to(device)
total_params = sum(p.numel() for p in teacher_model.parameters())
print(f'==> Number of parameters in Teacher: {cfg["teacher_model"]}: {total_params}')
print('==> Resuming from best checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
load_path = os.path.join('checkpoint', teacher_path)
checkpoint = torch.load(os.path.join(load_path, 'ckpt_best.pth'))
teacher_model.load_state_dict(checkpoint['net'])

student_model = get_model(**cfg['student_model'], num_classes=num_classes)
student_model = student_model.to(device)
total_params = sum(p.numel() for p in student_model.parameters())
print(f'==> Number of parameters in Student: {cfg["student_model"]}: {total_params}')
################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
# criterion = DynamicKDLoss(T=T, total_epochs=EPOCHS)
# criterion = CombinedLoss(T=T)
criterion = KDLoss(T=T, soft_target_loss_weight=soft_target_loss_weight, ce_loss_weight=ce_loss_weight)
# criterion = EntropyKDLoss(T=T, soft_target_loss_weight=soft_target_loss_weight, ce_loss_weight=ce_loss_weight)
# criterion = SoftTeacherKDLoss(T=T, soft_target_loss_weight=soft_target_loss_weight, ce_loss_weight=ce_loss_weight)
# criterion = KDCorrectLoss(T=T, soft_target_loss_weight=soft_target_loss_weight, ce_loss_weight=ce_loss_weight)
test_criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(student_model, opt_name, cfg['optimizer'])
if scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)])
elif scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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