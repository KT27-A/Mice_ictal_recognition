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
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from models.experimental import attempt_load
from detection.detect import detect
import sys
sys.path.insert(0, '/home/alien/zhangyujia/0613_mice_ictal_recognition/detection')

weights = '/home/alien/zhangyujia/0613_mice_ictal_recognition/detection/best.pt'
device = 'cuda:0'
# device = 'cpu'
detect_model = attempt_load(weights, map_location=device)
if device != 'cpu':
    detect_model = detect_model.half()

def center_crop(clip, model, sample_size=112):
    res = []
    # out = imageio.get_writer(save_path, fps=10)
    for i in range(len(clip)):
        if i % 64 == 0:
            if i + 64 > len(clip):
                x_center, _ = detect(clip[i], model)
            else:
                first_x_center, _ = detect(clip[i], model)
                mid_x_center, _ = detect(clip[i+31], model)
                end_x_center, _ = detect(clip[i+63], model)
                x_center = int((first_x_center + mid_x_center + end_x_center)/3)
            frame_height, frame_width, _ = clip[i].shape
            if not x_center:
                x_center = int(frame_width / 2)
            # x_start = x_center - int(sample_size/2)
            # x_end = x_center + int(sample_size/2)
            # fix = 30
            # if x_start < fix:
            #     x_start = fix
            #     x_end = fix + sample_size
                
            # if x_end > frame_width - fix:
            #     x_start = frame_width - sample_size - fix
            #     x_end = frame_width - fix
            if x_center < 44:
                x_start = 0
                x_end = 112
            elif x_center < 156:
                x_start = 43
                x_end = 155
            else:
                x_start = 87
                x_end = 199

        # res.append(np.transpose(clip[i][:, x_start:x_end, :] - [114.7748, 107.7354, 99.4750], [2, 1, 0]))
        cropped_clip = clip[i][:, x_start:x_end, :]
        # out.append_data(cropped_clip)
        res.append(np.transpose(cropped_clip - [114.7748, 107.7354, 99.4750], [2, 0, 1]))
        # res.append(np.transpose(cropped_clip - [99.4750, 107.7354, 114.7748], [2, 0, 1]))
        # res.append(np.transpose(cropped_clip - [0, 0, 0], [2, 0, 1]))
    return res


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
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip.append(frame)
                else:
                    cap.set(1, 0) # set the starting frame index 0
        else:
            while len(clip) < total_frames:
                (grabbed, frame) = cap.read()
                if grabbed:
                    frame = imutils.resize(frame, height=opt.sample_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip.append(frame)
    
    # pro_clip = center_crop(clip, detect_model)
    # # for i, crop in enumerate(pro_clip):
    # #     img = crop.transpose(1, 2, 0).astype(np.uint8)
    # #     cv2.imshow('img', clip[i])
    # #     cv2.imshow('img_2', img)
    # #     cv2.waitKey(0)

    # pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))

    
    pro_clip = cv2.dnn.blobFromImages(clip, 1.0,
                (opt.sample_size, opt.sample_size), (114.7748, 107.7354, 99.4750),
                        swapRB=True, crop=True)
    # for i, crop in enumerate(pro_clip):
    #     img = crop.transpose(1, 2, 0).astype(np.uint8)
    #     cv2.imshow('img', clip[i])
    #     cv2.imshow('img_2', img)
    #     cv2.waitKey(0)
    pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))
            
    
    
    # pro_clip = cv2.dnn.blobFromImages(clip, 1.0,
    #         (opt.sample_size, opt.sample_size), (0, 0, 0),
    #         swapRB=True, crop=True)
    # for i, crop in enumerate(pro_clip):
    #     img = crop.transpose(1, 2, 0).astype(np.uint8)
    #     cv2.imshow('img', clip[i])
    #     cv2.imshow('img_2', img)
    #     cv2.waitKey(0)
    # import pdb; pdb.set_trace()
    
    # pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))

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
            if self.train_val_test == 1:
                video_path = os.path.join(frame_dir, video_name)
            else:
                video_path = os.path.join(frame_dir, video_name+'.mp4')
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
            return scale_crop(clip, self.train_val_test, self.opt), label_id
        elif self.train_val_test == 1:
            clip = get_train_video(self.opt, frame_path, Total_frames)
            return scale_crop(clip, self.train_val_test, self.opt), label_id
        elif self.train_val_test == 2 or self.train_val_test == 3:
            video_path = frame_path
            clip = get_test_video_online(self.opt, video_path)
            return clip, label_id        
        


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
            # video_path = os.path.join(self.opt.frame_dir, video_name)
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
        
        return clip, label_id, video_path.split('/')[-1]