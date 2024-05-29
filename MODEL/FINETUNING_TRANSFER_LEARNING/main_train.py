import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pandas as pd
import numpy as np
import json
from random_state import set_seed
import time
from customize_dataset import load_finetune_dataset
from transfer_learning import transfer_learning
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

with open('./env_cam.json', 'r') as r:
    env_cam = json.load(r)
local_path = env_cam['local_dir']

# SET RANDOMSEED
set_seed(seed=42)

# Get Mean and Std about MAM10000
MeanStd = {"Mean": [0.7635212557080773, 0.5461279508434921, 0.5705303582621197], 
           "Std": [0.08962782189107416, 0.11830749629626626, 0.13295368820124384]}

print(f'IMPORTED....')

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description="TRAIN PARAMETERS.")
    
    # Define the possible choices for each argument
    parser.add_argument('--model', required=True, choices=['resnet18', 'resnet50', 'resnet101', 'resnet152'], help='Choose from: resnet18, resnet50, resnet101, resnet152')
    parser.add_argument('--epoch', required=True, type=int, choices=[1, 10, 50, 100], help='Choose from: 1, 10, 50, 100') # will be changed
    parser.add_argument('--augment', required=True, choices=['none', 'sample', 'whole'], help='Choose from: none, sample, whole')

    args = parser.parse_args()

    # Additional variables based on 'augment' and 'model' choice
     
    apply_augment = True if args.augment != 'none' else False
    apply_random = True if args.augment == 'sample' else False
    model_architecture = args.model
    num_epochs = args.epoch

    train_loader, valid_loader, test_loader, augment, randomness = load_finetune_dataset(
        data_dir=f'{local_path}/images/', 
        label_dir=f'{local_path}/HAM10000_label.csv',
        batch_size=64, MeanStd=MeanStd, num_workers=0, apply_augment=apply_augment, apply_random=apply_random,
        )

    print(f'DATA LOADER IS COMPLETED....\n')

    transfer_learning(
        model_architecture,
        num_epochs,
        train_loader, valid_loader, test_loader,
        apply_augment, augment, randomness
    )
    
    print(f'Transfer Learning Done.....')