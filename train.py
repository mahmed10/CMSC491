import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
import numpy as np

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm

import data

from models.unet import UNET
from models.deeplabv2 import get_deeplab_v2

from evaluate import iou_calculation
from domain_adaptation.train_UDA import train_domain_adaptation
from domain_adaptation.train_Sonly import train_source_only

from torchvision import transforms

parser = argparse.ArgumentParser(description='CMSC491')
parser.add_argument('--da_mode', type=str, default='withDA', help = 'withDA or withoutDA')
parser.add_argument('--source_dataset', type=str, default='cityscapes')
parser.add_argument('--target_dataset', type=str, default='semantickitti')
parser.add_argument('--train_mode', type=str, default='test')
parser.add_argument('--train_path_list', type=str, default='./dataset/synthia/trainlist.txt')
parser.add_argument('--val_path_list', type=str, default='./dataset/CityScapes/vallist.txt')
parser.add_argument('--data_mode', type=str, default='rgb')
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--image_width', type=int, default=572)
parser.add_argument('--image_height', type=int, default=572)
parser.add_argument('--num_classes', type=int, default=21)
parser.add_argument('--train_data_size', type=int, default=1500)
parser.add_argument('--model_path', type=str, default='./checkpoints/')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--load_model', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--start_epoches', type=int, default=72)
parser.add_argument('--learning_rate', type=float, default=0.0005)

args = parser.parse_args()



epoches = args.epoches

if torch.cuda.is_available():
	args.device = 'cuda:0'
	torch.cuda.empty_cache()
else:
	args.device = "cpu"


def main():
	args.source_loader = data.setup_loaders(args.source_dataset, args.train_path_list, args.batch_size)
	args.target_loader = data.setup_loaders(args.target_dataset, args.val_path_list, args.batch_size)
	print('Data Loaded Successfully!')

	args.model = UNET(in_channels=args.in_channels, classes=args.num_classes).to(args.device)
	if(args.da_mode == 'withDA'):
		train_domain_adaptation(args)
	if(args.da_mode == 'withoutDA'):
		train_source_only(args)

if __name__ == '__main__':
	main()