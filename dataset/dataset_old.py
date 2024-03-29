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
import cv2
import imutils

def get_test_video_online(opt, video_path):
    """
        Args:
            opt         : config options
            video_path  : video_path
        Returns:
            list(frames) : list of all video frames
        """

    clip = []
    i = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if opt.modality == 'RGB': 
        if total_frames < opt.sample_duration: 
            while len(clip) < opt.sample_duration:
                (grabbed, frame) = cap.read()
                if grabbed:
                    frame = imutils.resize(frame, height=opt.sample_size)
                    clip.append(frame)
                else:
                    cap.set(1, 0) # set the starting frame index 0
        else:
            while len(clip) < total_frames:
                (grabbed, frame) = cap.read()
                if grabbed:
                    frame = imutils.resize(frame, height=opt.sample_size)
                    clip.append(frame)
    
    pro_clip = cv2.dnn.blobFromImages(clip, 1.0,
            (opt.sample_size, opt.sample_size), (114.7748, 107.7354, 99.4750),
            swapRB=True, crop=False)
    pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))

    return pro_clip


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
            frame_dir = opt.frame_dir
        elif self.train_val_test == 1:
            filenames = opt.train_file
            frame_dir = opt.frame_dir
        elif self.train_val_test == 2:
            filenames = opt.val_file_1
            frame_dir = opt.val_path_1
        elif self.train_val_test == 3:
            filenames = opt.val_file_2
            frame_dir = opt.val_path_2


        f = open(os.path.join(self.opt.annotation_path, filenames), 'r')

        # for line in f:
        #     video_name, class_id = line.strip('\n').split(' #')
        #     video_path = os.path.join(self.opt.frame_dir, video_name)
        #     if os.path.exists(video_path) == True:
        #         self.data.append((video_path, class_id))
        #     else:
        #         print('ERROR no such video name {}'.format(video_name))
        for line in f:
            video_name, class_id = line.strip('\n').split(' #')
            video_path = os.path.join(frame_dir, video_name)
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
                    
        return scale_crop(clip, self.train_val_test, self.opt), label_id


class MICE_online(Dataset):
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
            filenames = opt.train_file
        elif self.train_val_test == 2:
            filenames = opt.val_file

        f = open(os.path.join(self.opt.annotation_path, filenames), 'r')

        for line in f:
            video_name, class_id = line.strip('\n').split(' #')
            video_path = os.path.join(self.opt.frame_dir, video_name+'.mp4')
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
        video_path = video[0]
        

        if self.train_val_test == 0: 
            clip = get_test_video_online(self.opt, video_path)
        
                    
        return clip, label_id