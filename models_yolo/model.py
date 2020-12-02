from __future__ import division
import torch
from torch import nn
from models import resnet
from models.resnet import get_fine_tuning_parameters
import os


def generate_model(config):
    model = resnet.resnet101(
            num_classes=config.getint('Network', 'classes'),
            shortcut_type=config.get('Network', 'resnet_shortcut'),
            cardinality=config.getint('Network', 'resnet_cardinality'),
            sample_size=config.getint('Network', 'sample_size'),
            sample_duration=config.getint('Network', 'sample_duration'),
            input_channels=config.getint('Network', 'input_channels'),
            )
    if config.getboolean('Network', 'use_cuda'):
        model = model.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = nn.DataParallel(model)
    pretrain_path = config.get('Network', 'pretrain_path')
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)
        
        assert config.get('Network', 'arch') == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'])
        model.module.fc = nn.Linear(model.module.fc.in_features, config.getint('Network', 'n_finetune_classes'))
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, config.getint('Network', 'ft_begin_index'))
        return model, parameters

    return model, model.parameters()

