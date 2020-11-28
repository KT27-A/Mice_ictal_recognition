from dataset.dataset_val_min import *
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
from dataset.utils import *
    

def test(opts):
    if torch.cuda.is_available():
        opts.cuda = True
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)

    print("Preprocessing testing data ...")
    test_data = globals()['{}'.format(opts.dataset)](split = opts.split, train = 0, 
    opt = opts)
    print("Length of testing data = ", len(test_data))
    
    if opts.modality=='RGB': opts.input_channels = 3
    elif opts.modality=='Flow': opts.input_channels = 2

    print("Preparing datatloaders ...")
    test_dataloader = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, 
    num_workers=opts.n_workers, pin_memory=True, drop_last=False)
    print("Length of validation datatloader = ",len(test_dataloader))
    
    # Loading model and checkpoint
    model, parameters = generate_model(opts)

    accuracies = AverageMeter()

    #Path to store results
    result_path = "{}/{}/".format(opts.result_path, opts.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)    

    if opts.log:
        f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}_online_{}_{}".format(opts.model, 
        opts.model_depth, opts.dataset, opts.split, opts.modality, opts.sample_duration, 
        opts.test_file, opts.resume_path1.split('/')[1])), 'w+')
        f.write(str(opts))
        f.write('\n')
        f.flush()
        prob_list = open(os.path.join(result_path, "prob_FP_{}{}_{}_{}_{}_{}_online_{}_{}".format(opts.model, 
        opts.model_depth, opts.dataset, opts.split, opts.modality, opts.sample_duration, 
        opts.test_file, opts.resume_path1.split('/')[1])), 'w+')
    
    softmax = nn.Softmax(dim=1)
    
    model.eval()
    with torch.no_grad():   
        for i, (clip, targets, video_name) in enumerate(test_dataloader):
            clip = torch.squeeze(clip)
            if opts.modality == 'RGB':
                inputs = torch.Tensor(int(clip.shape[1]/opts.sample_duration)+1, 3, 
                    opts.sample_duration, opts.sample_size, opts.sample_size)

            for k in range(inputs.shape[0]-1):
                inputs[k, :, :, :, :] = clip[:,k*opts.sample_duration:(k+1)*opts.sample_duration,:,:]
            
            inputs[-1, :, :, :, :] = clip[:, -opts.sample_duration:, :, :]
            
            if opts.cuda:
                inputs = inputs.cuda()

            outputs = model(inputs)
            pre_label = torch.sum(outputs.topk(1)[1]).item()

            prob_outputs = softmax(outputs)
            if targets.item() == 0:
                if pre_label > 0:
                    acc = 0
                    line = 'False Positive: name={}, prob={}'.format(video_name, prob_outputs)
                    prob_list.write(line+'\n')
                    prob_list.flush()
                else:
                    acc = 1
                # print(prob_outputs)
                # if pre_label > 1:
                #     for h in range(10):
                #         if 0 < prob_outputs[h][1] - prob_outputs[h][0] < 0.3:
                #             # if prob of case bigger than control and their gap smaller than 0.3, it should be control
                #             pre_label = pre_label - 1
                #     if pre_label > 1:
                #         # acc = 0 means it's a case
                #         acc = 0
                #     else:
                #         acc = 1
                # else:
                #     acc = 1
            else:
                if pre_label > 0:
                    acc = 1
                else:
                    acc = 0
            
            accuracies.update(acc, inputs.size(0))       
            
            line = "Video[" + str(i) + "] :  "  + "\t predict = " + str(pre_label) + \
                "\t true = " +str(int(targets[0])) + "\t acc = " + str(accuracies.avg)
                    
            print(line)
            # print(outputs.topk(1)[1])
            # print(prob_outputs)
            if opts.log:
                f.write(line + '\n')
                f.flush()
    
    print("Video accuracy = ", accuracies.avg)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'
    if opts.log:
        f.write(line)

if __name__=="__main__":
    # print configuration options
    opts = parse_opts()
    print(opts)
    t_file = opts.test_file.split(',')
    if len(t_file) > 1:
        for file in t_file:
            opts.test_file = file
            test(opts)
    else:
        test(opts)
            

    
    
