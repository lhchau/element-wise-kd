import torch
import os
from .utils import *
from .bypass_bn import *

def loop_one_epoch_sam_kl(
    dataloader,
    student,
    criterion,
    optimizer,
    device,
    logging_dict
    ):
    loss = 0
    total = 0
    correct = 0 
    
    student.train()
    ce_criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        enable_running_stats(student)  # <- this is the important line
        outputs = student(inputs)
        optimizer.zero_grad()
        first_loss = ce_criterion(outputs, targets)
        first_loss.backward(retain_graph=True)        
        optimizer.first_step(zero_grad=True)
        
        disable_running_stats(student)  # <- this is the important line
        with torch.no_grad():
            outputs_no_grad = outputs.clone().detach()
        criterion(outputs_no_grad, student(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)
        
        if (batch_idx + 1) == len(dataloader):
            logging_dict.update(get_norm(optimizer))
            logging_dict['Train/ce_loss'] = criterion.ce_loss
            logging_dict['Train/kl_loss'] = criterion.kl_loss
                    
        with torch.no_grad():
            loss += float(torch.mean(first_loss).item())
            loss_mean = loss/(batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
    logging_dict[f'Train/loss'] = loss_mean
    logging_dict[f'Train/acc'] = acc
    

def loop_one_epoch_knowledge_distillation(
    dataloader,
    student,
    teacher,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch
    ):
    loss = 0
    total = 0
    correct = 0 
    
    teacher.eval()
    student.train()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        opt_name = type(optimizer).__name__
        criterion_name = type(criterion).__name__
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        if opt_name == 'SGD':
            outputs = student(inputs)
            
            if (batch_idx + 1) != len(dataloader):
                first_loss = criterion(teacher_logits, outputs, targets, mode='normal')
                first_loss.backward()
            else:
                first_loss = criterion(teacher_logits, outputs, targets, mode='ce')
                first_loss.backward(retain_graph=True)
                ce_grads = [p.grad.clone() for p in student.parameters()]
                second_loss = criterion(teacher_logits, outputs, targets, mode='kd')
                second_loss.backward()
                total_grads = [p.grad.clone() for p in student.parameters()]
                kd_grads = [to - ce for ce, to in zip(ce_grads, total_grads)]
                
                masksA, masksB, masksC1, masksC2 = [], [], [], []
                groupA, groupB, groupC1, groupC2 = 0, 0, 0, 0
                meanA, meanB, meanC1, meanC2 = [], [], [], []
                for ce, kd in zip(ce_grads, kd_grads):
                    ratio = ce / (kd + 1e-12)
                    maskA = ratio >= 1
                    maskB = torch.logical_and(ratio < 1, ratio >= 0)
                    maskC1 = torch.logical_and(ratio > -1, ratio < 0)
                    maskC2 = ratio <= -1
                    masksA.append(maskA)
                    masksB.append(maskB)
                    masksC1.append(maskC1)
                    masksC2.append(maskC2)
                    groupA += torch.sum(maskA)
                    groupB += torch.sum(maskB)
                    groupC1 += torch.sum(maskC1)
                    groupC2 += torch.sum(maskC2)
                    meanA.append(torch.mean(ratio[maskA]))
                    meanB.append(torch.mean(ratio[maskB]))
                    meanC1.append(torch.mean(ratio[maskC1]))
                    meanC2.append(torch.mean(ratio[maskC2]))
                logging_dict['Stat/groupA'] = groupA
                logging_dict['Stat/groupB'] = groupB
                logging_dict['Stat/groupC1'] = groupC1
                logging_dict['Stat/groupC2'] = groupC2
                logging_dict['Stat/meanA'] = torch.mean(torch.tensor(meanA))
                logging_dict['Stat/meanB'] = torch.mean(torch.tensor(meanB))
                logging_dict['Stat/meanC1'] = torch.mean(torch.tensor(meanC1))
                logging_dict['Stat/meanC2'] = torch.mean(torch.tensor(meanC2))
                
                layer_names = [name for name, param in student.named_parameters() if param.requires_grad]
                sim_with_bias, sim_without_bias = get_cosine_similarity(ce_grads, kd_grads, layer_names)
                logging_dict['Info/sim_with_bias'] = sim_with_bias
                logging_dict['Info/sim_without_bias'] = sim_without_bias
                logging_dict['Info/ce_grad_norm'] = get_grad_norm(ce_grads)
                logging_dict['Info/kd_grad_norm'] = get_grad_norm(kd_grads)
                logging_dict['Info/total_grad_norm'] = get_grad_norm(total_grads)
                logging_dict['Info/ratio_grad_norm'] = logging_dict['Info/ce_grad_norm'] / logging_dict['Info/kd_grad_norm']
                
            # first_loss = criterion(teacher_logits, outputs, targets, mode='ce')
            # first_loss.backward(retain_graph=True)
            # ce_grads = [p.grad.clone() for p in student.parameters()]
            # second_loss = criterion(teacher_logits, outputs, targets, mode='kd')
            # second_loss.backward()
            # total_grads = [p.grad.clone() for p in student.parameters()]
            # kd_grads = [to - ce for ce, to in zip(ce_grads, total_grads)]
            
            # masksA, masksB, masksC1, masksC2, masksC = [], [], [], [], []
            # groupA, groupB, groupC1, groupC2 = 0, 0, 0, 0
            # meanA, meanB, meanC1, meanC2 = [], [], [], []
            # for ce, kd in zip(ce_grads, kd_grads):
            #     ratio = ce / (kd + 1e-6)
            #     maskA = ratio >= 1
            #     maskB = torch.logical_and(ratio < 1, ratio >= 0)
            #     maskC1 = torch.logical_and(ratio > -1, ratio < 0)
            #     maskC2 = ratio <= -1
            #     maskC = ratio < 0
            #     masksA.append(maskA)
            #     masksB.append(maskB)
            #     masksC.append(maskC)
            #     masksC1.append(maskC1)
            #     masksC2.append(maskC2)
            #     groupA += torch.sum(maskA)
            #     groupB += torch.sum(maskB)
            #     groupC1 += torch.sum(maskC1)
            #     groupC2 += torch.sum(maskC2)
            #     meanA.append(torch.mean(ratio[maskA]))
            #     meanB.append(torch.mean(ratio[maskB]))
            #     meanC1.append(torch.mean(ratio[maskC1]))
            #     meanC2.append(torch.mean(ratio[maskC2]))
                
            # if (batch_idx + 1) == len(dataloader):
            #     logging_dict['Stat/groupA'] = groupA
            #     logging_dict['Stat/groupB'] = groupB
            #     logging_dict['Stat/groupC1'] = groupC1
            #     logging_dict['Stat/groupC2'] = groupC2
            #     logging_dict['Stat/meanA'] = torch.mean(torch.tensor(meanA))
            #     logging_dict['Stat/meanB'] = torch.mean(torch.tensor(meanB))
            #     logging_dict['Stat/meanC1'] = torch.mean(torch.tensor(meanC1))
            #     logging_dict['Stat/meanC2'] = torch.mean(torch.tensor(meanC2))
                
            #     layer_names = [name for name, param in student.named_parameters() if param.requires_grad]
            #     sim_with_bias, sim_without_bias = get_cosine_similarity(ce_grads, kd_grads, layer_names)
            #     logging_dict['Info/sim_with_bias'] = sim_with_bias
            #     logging_dict['Info/sim_without_bias'] = sim_without_bias
            #     logging_dict['Info/ce_grad_norm'] = get_grad_norm(ce_grads)
            #     logging_dict['Info/kd_grad_norm'] = get_grad_norm(kd_grads)
            #     logging_dict['Info/total_grad_norm'] = get_grad_norm(total_grads)
            #     logging_dict['Info/ratio_grad_norm'] = logging_dict['Info/ce_grad_norm'] / logging_dict['Info/kd_grad_norm']
                    
            # with torch.no_grad():
            #     for p, m, ce, kd in zip(student.parameters(), masksC2, ce_grads, kd_grads):
            #         dest_grad = ce * 2
            #         p.grad = dest_grad.mul(m) + p.grad.mul(torch.logical_not(m))
            #         # p.grad = p.grad.mul(torch.logical_not(m)) + kd.mul(m).mul(8)
            
            optimizer.step()
            optimizer.zero_grad()
            if (batch_idx + 1) == len(dataloader):
                try:
                    logging_dict['Train/ce_loss'] = criterion.ce_loss
                    logging_dict['Train/kl_loss'] = criterion.kl_loss
                except:
                    pass
        else:
            enable_running_stats(student)  # <- this is the important line
            outputs = student(inputs)
            optimizer.zero_grad()
            first_loss = criterion(teacher_logits, outputs, targets)
            first_loss.backward(retain_graph=True)        
            optimizer.first_step(zero_grad=True)
            
            disable_running_stats(student)  # <- this is the important line
            criterion(teacher_logits, student(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
            
            if (batch_idx + 1) == len(dataloader):
                logging_dict.update(get_norm(optimizer))
                if criterion_name == 'KDLoss':
                    logging_dict['Train/ce_loss'] = criterion.ce_loss
                    logging_dict['Train/kl_loss'] = criterion.kl_loss
                    
        with torch.no_grad():
            loss += float(torch.mean(first_loss).item())
            loss_mean = loss/(batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
    logging_dict[f'Train/loss'] = loss_mean
    logging_dict[f'Train/acc'] = acc
    

def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type='train',
    logging_name=None,
    best_acc=0,
    ):
    loss = 0
    total = 0
    correct = 0
    
    if loop_type == 'train': 
        net.train()
        idx_to_class = dataloader.dataset.classes
        num_classes = len(idx_to_class)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            opt_name = type(optimizer).__name__
            if opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                torch.mean(first_loss).backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs1 = net(inputs)
                outputs = outputs1
                optimizer.zero_grad()
                first_loss = criterion(outputs1, targets)
                first_loss.backward(retain_graph=True)        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                
                if (batch_idx + 1) == len(dataloader):
                    logging_dict.update(get_norm(optimizer))
                    try:
                        logging_dict['Train/ce_loss'] = criterion.ce_loss
                        logging_dict['Train/kl_loss'] = criterion.kl_loss
                    except:
                        pass
                        
            with torch.no_grad():
                loss += float(torch.mean(first_loss).item())
                loss_mean = loss/(batch_idx+1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
                
    elif loop_type == 'test':
        net.eval()
        idx_to_class = dataloader.dataset.classes
        num_classes = len(idx_to_class)
        class_correct = [0] * num_classes  # Correct predictions per class
        class_total = [0] * num_classes    # Total samples per class
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(torch.mean(first_loss).item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Class-wise accuracy
                for i in range(len(targets)):
                    label = targets[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
            if acc > best_acc:
                print('Saving best checkpoint ...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'loss': loss,
                    'epoch': epoch
                }
                save_path = os.path.join('checkpoint', logging_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
                best_acc = acc
            logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
        logging_dict[f'{loop_type.title()}/gen_gap'] = logging_dict['Train/acc'] - acc
        for idx, (c, t) in enumerate(zip(class_correct, class_total)):
            logging_dict[f'classes/{idx}.{idx_to_class[idx]}'] = 100. * c / t
    else:
        # Load checkpoint.
        print('==> Resuming from best checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        save_path = os.path.join('checkpoint', logging_name)
        checkpoint = torch.load(os.path.join(save_path, 'ckpt_best.pth'))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total

                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
                
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'test': 
        return best_acc, acc