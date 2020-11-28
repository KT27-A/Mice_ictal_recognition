from dataset.dataset_val_min import *
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import *
from loss.focal_loss import FocalLoss


if __name__=="__main__":
    opts = parse_opts()
    print(opts)
    
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)
    torch.manual_seed(opts.manual_seed)

    print("Preprocessing train data ...")
    train_data = globals()['{}'.format(opts.dataset)](split = opts.split, train = 1, opt = opts)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data = globals()['{}'.format(opts.dataset)](split = opts.split, train = 2, opt = opts)
    print("Length of validation data = ", len(val_data))
    
    if opts.modality=='RGB': opts.input_channels = 3
    elif opts.modality=='Flow': opts.input_channels = 2

    print("Preparing dataloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opts.batch_size, shuffle=True, num_workers = opts.n_workers, pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = 1, shuffle=True, num_workers = opts.n_workers, pin_memory = True, drop_last=False)
    print("Length of train dataloader = ",len(train_dataloader))
    print("Length of validation dataloader = ",len(val_dataloader))    
   
    # define the model 
    print("Loading model... ", opts.model, opts.model_depth)
    model, parameters = generate_model(opts)
    
    criterion = FocalLoss(alpha=[0.8, 0.4], gamma=2, criterion=nn.CrossEntropyLoss(reduction='none').cuda()).cuda()
    
    log_path = os.path.join(opts.result_path, opts.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    if opts.log == 1:
        if opts.resume_path1:
            begin_epoch = int(opts.resume_path1.split('/')[-1].split('_')[1])
            epoch_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'
                        .format(opts.dataset, opts.sample_duration, opts.model, opts.model_depth))
                        ,['epoch', 'loss', 'acc', 'lr'], overlay=False)
            val_logger   = Logger(os.path.join(log_path, '{}_val_clip{}model{}{}.log'
                            .format(opts.dataset, opts.sample_duration, opts.model, opts.model_depth))
                            ,['epoch', 'loss', 'acc'], overlay=False)
        else:
            begin_epoch = 0
            epoch_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'
                        .format(opts.dataset, opts.sample_duration, opts.model, opts.model_depth))
                        ,['epoch', 'loss', 'acc', 'lr'], overlay=True)
            val_logger   = Logger(os.path.join(log_path, '{}_val_1_clip{}model{}{}.log'
                            .format(opts.dataset, opts.sample_duration, opts.model, opts.model_depth))
                            ,['epoch', 'loss', 'acc'], overlay=True)
            if opts.val_file_2:
                val_logger_2   = Logger(os.path.join(log_path, '{}_val_2_clip{}model{}{}.log'
                            .format(opts.dataset, opts.sample_duration, opts.model, opts.model_depth))
                            ,['epoch', 'loss', 'acc'], overlay=True)
            
           
    print("Initializing the optimizer ...")
    if opts.pretrain_path: 
        opts.weight_decay = 1e-5
        opts.learning_rate = 0.001
        # opts.weight_decay = 5e-4
        # opts.learning_rate = 0.1

    if opts.nesterov: dampening = 0
    else: dampening = opts.dampening
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opts.learning_rate, opts.momentum, dampening, opts. weight_decay, opts.nesterov))
    print("LR patience = ", opts.lr_patience)
    
    
    optimizer = optim.SGD(
        parameters,
        lr=opts.learning_rate,
        momentum=opts.momentum,
        dampening=dampening,
        weight_decay=opts.weight_decay,
        nesterov=opts.nesterov)


    if opts.resume_path1 != '':
        optimizer.load_state_dict(torch.load(opts.resume_path1)['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.lr_patience)

    print('run')
    for epoch in range(begin_epoch, opts.n_epochs + 1):
        
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            outputs = model(inputs)

            # targets_m = targets.float().clone()
            # for i in range(opts.batch_size):
            #     if targets[i] == 0:
            #         targets_m[i] = 0.214
            #     else:
            #         targets_m[i] = 1
            # import pdb; pdb.set_trace()
            # outputs_m = targets_m*outputs
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Lr {lr}'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies,
                      lr=optimizer.param_groups[-1]['lr']))

        if opts.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[-1]['lr']
            })

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        print("Length of validation data = ", len(val_data))
        with torch.no_grad():
            for i, (clip, targets) in enumerate(val_dataloader):
                clip = torch.squeeze(clip)
                if opts.modality == 'RGB':
                    inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 3, opts.sample_duration, opts.sample_size, opts.sample_size)
                elif opts.modality == 'Flow':
                    inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 2, opts.sample_duration, opts.sample_size, opts.sample_size)

                for k in range(inputs.shape[0]-1):
                    inputs[k, :, :, :, :] = clip[:,k*opts.sample_duration:(k+1)*opts.sample_duration,:,:]
                
                inputs[-1, :, :, :, :] = clip[:, -opts.sample_duration:, :, :]
                
                inputs = inputs.cuda()

                outputs = model(inputs)
                pre_label = torch.sum(outputs.topk(1)[1]).item()
            
                if targets.item() == 0:
                    if pre_label > 0:
                        acc = 0
                    else:
                        acc = 1
                else:
                    if pre_label > 0:
                        acc = 1
                    else:
                        acc = 0
                                
                accuracies.update(acc, 1)

                line = "Video[" + str(i) + "] :  "  + "\t predict = " + str(pre_label) + "\t true = " +str(int(targets[0])) + "\t acc = " + str(accuracies.avg)
                print(line)
        

        accuracy_val = accuracies.avg
        if accuracy_val > list(opts.highest_val.values())[0]:
            old_key = list(opts.highest_val.keys())[0]
            file_path = os.path.join(opts.result_path, old_key)
            if os.path.exists(file_path):
                os.remove(file_path)
            opts.highest_val.pop(old_key)
            opts.highest_val['save_{}_max.pth'.format(epoch)] = accuracy_val

            save_file_path = os.path.join(opts.result_path,
                                        'save_{}_max.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opts.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
        if epoch % 1 == 0:
            save_file_path = os.path.join(opts.result_path,
                                        'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opts.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        if opts.log == 1:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        if opts.val_file_2:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            val_data_2 = globals()['{}'.format(opts.dataset)](split=opts.split, train=3, opt=opts)
            val_dataloader_2 = DataLoader(val_data_2, batch_size = 1, 
            shuffle=False, num_workers = opts.n_workers, pin_memory = True, drop_last=False)
            print("Length of validation_2 data = ", len(val_data_2))
            with torch.no_grad():
                for i, (clip, targets) in enumerate(val_dataloader_2):
                    clip = torch.squeeze(clip)
                    if opts.modality == 'RGB':
                        inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 3, opts.sample_duration, opts.sample_size, opts.sample_size)
                    elif opts.modality == 'Flow':
                        inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 2, opts.sample_duration, opts.sample_size, opts.sample_size)

                    for k in range(inputs.shape[0]-1):
                        inputs[k, :, :, :, :] = clip[:,k*opts.sample_duration:(k+1)*opts.sample_duration,:,:]
                    
                    inputs[-1, :, :, :, :] = clip[:, -opts.sample_duration:, :, :]
                    
                    inputs = inputs.cuda()

                    outputs = model(inputs)
                    pre_label = torch.sum(outputs.topk(1)[1]).item()
                
                    if targets.item() == 0:
                        if pre_label > 0:
                            acc = 0
                        else:
                            acc = 1
                    else:
                        if pre_label > 0:
                            acc = 1
                        else:
                            acc = 0
                                    
                    accuracies.update(acc, 1)

                    line = "Video[" + str(i) + "] :  "  + "\t predict = " + str(pre_label) + "\t true = " +str(int(targets[0])) + "\t acc = " + str(accuracies.avg)
                    print(line)
            
            val_logger_2.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        


        



