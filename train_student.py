import torch
import datetime
import pprint

from models import *
from utils import *
from dataloaders import *
from optimizers import *

################################
#### 0. SETUP CONFIGURATION
################################
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
cfg = exec_configurator()

if cfg['student_model']['model_name'] in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2'] and cfg['dataloader']['data_name'] == 'cifar100':
    cfg['optimizer']['lr'] = 0.01
    
seed = cfg['trainer'].get('seed', 42)
initialize(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, logging_dict = 0, {}
EPOCHS = cfg['trainer']['epochs'] 
teacher_path = cfg['trainer'].get('teacher_path', None)

kd_type = cfg['trainer'].get('kd_type', 'kd')
T = cfg['trainer'].get('T', 4)
kl_weight = cfg['trainer'].get('kl_weight', 1)
ce_weight = cfg['trainer'].get('ce_weight', 1)
rho = cfg['trainer'].get('rho', None)
dkd_beta = cfg['trainer'].get('dkd_beta', None)

################################
#### 0.b LOGGING
################################
print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += (f'_T={T}' + f'_kl_w={kl_weight}' + f'_ce_w={ce_weight}' + f'_kd_t={kd_type}' + f'_rho={rho}' + f'_seed={seed}' + f'_{current_time}')
if kd_type == 'dkdadakd':
    logging_name += f'_dkd_be={dkd_beta}'

project_name = cfg['logging']['project_name']
log_dir = os.path.join("logs", project_name, logging_name)
txt_logger = TextLogger(log_dir=log_dir, filename="train_log.txt")

txt_logger.log_config(cfg)
txt_logger.log_event(f"device={device}")
txt_logger.log_event(f"teacher_path={teacher_path}")
txt_logger.log_event(f"logging_name={logging_name}")

pprint.pprint(cfg)
################################
#### 1. BUILD THE DATASET
################################
train_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])

################################
#### 2. BUILD THE NEURAL NETWORK
################################
teacher_model_name = cfg['teacher_model'].pop('model_name', None)
student_model_name = cfg['student_model'].pop('model_name', None)

if cfg['dataloader']['data_name'] == "imagenet":
    teacher_model = imagenet_model_dict[teacher_model_name](pretrained=True).to(device)
    student_model = imagenet_model_dict[student_model_name](pretrained=False).to(device)
else:
    teacher_model = model_dict[teacher_model_name](num_classes=num_classes, **cfg['teacher_model']).to(device)
    
    msg = f'==> Load checkpoint for Teacher: {teacher_model_name}'
    print(msg)
    txt_logger.write(msg)
    
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(teacher_path, weights_only=False)
    teacher_model.load_state_dict(checkpoint['model'])

    student_model = model_dict[student_model_name](num_classes=num_classes, **cfg['student_model']).to(device)
    
    total_params = sum(p.numel() for p in student_model.parameters())
    msg = f'==> Number of parameters in Student: {student_model_name}: {total_params}'
    print(msg)
    txt_logger.write(msg)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
if kd_type == 'kd':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight)
elif kd_type == 'ablation':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight, mode=kd_type)
elif kd_type == 'ablation_add':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight, mode=kd_type)
elif kd_type == 'ablation2':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight, mode=kd_type)
elif kd_type == 'ablation_add2':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight, mode=kd_type)
elif kd_type == 'dTs':
    criterion = KD(T=T, kl_weight=kl_weight, ce_weight=ce_weight, mode=kd_type)
elif kd_type == 'ats':
    criterion = ATS(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho, warmup=warmup)
elif kd_type == 'dkdadakd':
    criterion = DKDADAKD(kl_weight=kl_weight, ce_weight=ce_weight, rho=rho, dkd_beta=dkd_beta, warmup=warmup)

test_criterion = torch.nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(student_model, opt_name, cfg['optimizer'])
if cfg['dataloader']['data_name'] == "imagenet":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
else:
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
            
        text_logger.log_dict(epoch, logging_dict)