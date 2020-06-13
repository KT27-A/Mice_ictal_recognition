from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
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
import torch.utils
import sys
from utils import *
    

if __name__=="__main__":
    # print configuration options
    opts = parse_opts()
    print(opts)
    if torch.cuda.is_available():
        opts.cuda = True
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)

    print("Preprocessing testing data ...")
    test_data = globals()['{}'.format(opts.dataset)](split = opts.split, train = 0, opt = opts)
    print("Length of testing data = ", len(test_data))
    
    if opts.modality=='RGB': opts.input_channels = 3
    elif opts.modality=='Flow': opts.input_channels = 2

    print("Preparing datatloaders ...")
    test_dataloader = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_workers, pin_memory=True, drop_last=False)
    print("Length of validation datatloader = ",len(test_dataloader))
    
    # Loading model and checkpoint
    model, parameters = generate_model(opts)
    criterion_rl = nn.CrossEntropyLoss(reduction='none').cuda() 
    accuracies = AverageMeter()
    clip_accuracies = AverageMeter()
    
    #Path to store results
    result_path = "{}/{}/".format(opts.result_path, opts.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)    

    if opts.log:
        f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}_plusone.txt".format(opts.model, opts.model_depth, opts.dataset, opts.split, opts.modality, opts.sample_duration)), 'w+')
        f.write(str(opts))
        f.write('\n')
        f.flush()
    if opts.resume_path1:
        print('loading checkpoint {}'.format(opts.resume_path1))
        checkpoint = torch.load(opts.resume_path1)
        assert opts.arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():   
        for i, (clip, targets) in enumerate(test_dataloader):
            clip = torch.squeeze(clip)
            if opts.cuda:
                targets = targets.cuda(non_blocking=True)
            if opts.modality == 'RGB':
                inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 3, opts.sample_duration, opts.sample_size, opts.sample_size)
            elif opts.modality == 'Flow':
                inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 2, opts.sample_duration, opts.sample_size, opts.sample_size)

            for k in range(inputs.shape[0]-1):
                inputs[k, :, :, :, :] = clip[:,k*opts.sample_duration:(k+1)*opts.sample_duration,:,:]
            
            inputs[-1, :, :, :, :] = clip[:, -opts.sample_duration:, :, :]
            
            if opts.cuda:
                inputs = inputs.cuda()

            outputs_var = model(inputs)

            pred = np.array(torch.mean(outputs_var, dim=0, keepdim=True).topk(1)[1].cpu().data[0])

            acc = float(pred[0] == targets[0])
                            
            accuracies.update(acc, 1)            
            
            line = "Video[" + str(i) + "] :  "  + "\t top1 = " + str(pred[0]) +  "\t true = " +str(int(targets[0])) + "\t video = " + str(accuracies.avg)
            print(line)
            if opts.log:
                f.write(line + '\n')
                f.flush()
    
    print("Video accuracy = ", accuracies.avg)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'
    if opts.log:
        f.write(line)
    
