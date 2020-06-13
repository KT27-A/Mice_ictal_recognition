from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from .preprocess_data import *
from PIL import Image, ImageFilter
import pickle
import glob
#import dircache
import pdb


def get_test_video(opt, frame_path, Total_frames):
    """
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames
        """

    clip = []
    i = 0
    loop = 0
    if Total_frames < opt.sample_duration: loop = 1
    
    if opt.modality == 'RGB': 
        while len(clip) < max(opt.sample_duration, Total_frames):
            
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg'%(i+1)))
                clip.append(im.copy())
                im.close()
            except:
                print('ERROR no such image {}'.format(os.path.join(frame_path, '%05d.jpg'%(i+1))))
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0

    elif opt.modality == 'Flow':  
        while len(clip) < 2*max(opt.sample_duration, Total_frames):
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(i+1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
                
    elif  opt.modality == 'RGB_Flow':
        while len(clip) < 3*max(opt.sample_duration, Total_frames):
            try:
                im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(i+1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(i+1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    return clip
    

def get_train_video(opt, frame_path, Total_frames):
    """
        Chooses a random clip from a video for training/ validation
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip (list of frames of length sample_duration) from a video for training/ validation
        """
    clip = []
    i = 0
    loop = 0

    # choosing a random frame
    if Total_frames <= opt.sample_duration: 
        loop = 1
        start_frame = 0
    else:
        start_frame = np.random.randint(0, Total_frames - opt.sample_duration)
    
    if opt.modality == 'RGB': 
        while len(clip) < opt.sample_duration:
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg'%(start_frame+i+1)))
                clip.append(im.copy())
                im.close()
            except:
                print('ERROR no such image {}'.format(os.path.join(frame_path, '%05d.jpg'%(i+1))))
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0

    elif opt.modality == 'Flow':  
        while len(clip) < 2*opt.sample_duration:
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(start_frame+i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(start_frame+i+1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
                
    elif  opt.modality == 'RGB_Flow':
        while len(clip) < 3*opt.sample_duration:
            try:
                im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(start_frame+i+1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(start_frame+i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(start_frame+i+1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    return clip


class MICE(Dataset):
    """MICE Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation 
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt
        
        with open(os.path.join(self.opt.annotation_path, "class.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]
        
        with open(os.path.join(self.opt.annotation_path, "class.txt")) as lab_file:
            index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 2

        # indexes for training/test set

        self.data = []        # (filename , lab_id)
        if self.train_val_test == 0:
            filenames = opt.test_file
        elif self.train_val_test == 1:
            filenames = 'train_mice.txt'
        elif self.train_val_test == 2:
            filenames = 'test_mice.txt'

        f = open(os.path.join(self.opt.annotation_path, filenames), 'r')

        for line in f:
            video_name, class_id = line.strip('\n').split(' #')
            video_path = os.path.join(self.opt.frame_dir, video_name)
            if os.path.exists(video_path) == True:
                self.data.append((video_path, class_id))
            else:
                print('ERROR no such video name {}'.format(video_name))
        f.close()
    def __len__(self):
        '''
        returns number of test set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = int(video[1])
        frame_path = video[0]
        Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))

        if self.train_val_test == 0: 
            clip = get_test_video(self.opt, frame_path, Total_frames)
        else:
            clip = get_train_video(self.opt, frame_path, Total_frames)
                    
        return((scale_crop(clip, self.train_val_test, self.opt), label_id))