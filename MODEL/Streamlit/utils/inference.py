'''
TO DO : Code refactoring and annotations
'''

# IMPORT LIBRARY
import torch
import os
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd
import json
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM 
import tempfile
import warnings
from utils.random_state import set_seed
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')

def Get_label_mapping(local_dir:str) -> (int, dict):
    '''Get label information from original data'''
    
    label_dir= local_dir + 'HAM10000_label.csv'
    label_df = pd.read_csv(label_dir)
    # Convert DataFrame to dictionary
    label_dic_list = label_df.to_dict('records')
    label_dic = {row['image_id']: row['label'] for row in label_dic_list}

    num_classes = len(label_df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(label_df.iloc[:, 1].unique())}
    return num_classes, label_mapping, label_dic

def Set_pretrained_model(num_classes:int, pretrained_model_path:str) -> (torch.nn.Module, str):
    '''Use Pretrained Model to use in the service
    Parameters
        - num_classes : Target Y Label Number which get from Get_label_mapping
        - pretrained_model_path : Saved local model path
    Returns
        tuple: A tuple containing:
        - model : pretrained model
        - model_architecture : Model architecture such as resnet, vgg and etc
    '''
    pretrained_state_dict = torch.load(pretrained_model_path)
    model_architecture = pretrained_model_path.split('/')[-1].split('_')[0]

    # Instantiate the model
    try:
        # Dynamically load the model class
        model_class = getattr(models, model_architecture)
        model = model_class(weights=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(pretrained_state_dict)

    except Exception as e :
        raise ValueError(f'Please check model architecture name which called "{model_architecture}"')
    return model, model_architecture

def Get_MeanStd_value(local_dir:str):
    # Get Mean and Std about MAM10000
    with open(local_dir + 'HAM10000_MeanStd.json', 'r') as ms_json:
        MeanStd = json.load(ms_json)
        
    transform = transforms.Compose([
        transforms.ToTensor()
        , transforms.Resize((224, 224), antialias=True) # ImageNet Trained size
        , transforms.Normalize(mean=MeanStd['Mean'], std=MeanStd['Std'])
    ])
    return transform

def inference(target_image, model, model_architecture, device, transform, label_mapping, label_dic):
    set_seed(seed=42)
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, target_image.name)
    target_name  = target_image.name.split(".")[0]
    target_answer= label_dic[target_name]

    with open(img_path, 'wb') as img_file:
        img_file.write(target_image.getvalue())
        
    if 'resnet' in model_architecture:
        target_layer = [model.layer4]
        model.eval()
        model.to(device)
        
        # Sample image
        test_image = cv2.imread(img_path, 1)[:, :, ::-1]
        test_image = np.float32(test_image) / 255
        
        # Resize the target_image to match the expected shape
        test_image = cv2.resize(test_image, (224, 224))
        
        # Apply the transformation to the sample image
        input_image = transform(test_image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_image)
            
        # Get the predicted class
        predicted_class = torch.argmax(output).item()

        # Convert predicted class index to label
        predicted_label = next(key for key, value in label_mapping.items() if value == predicted_class)

        # Predicted value
        targets = None
        
        # Construct the CAM object once, and then re-use it on many images:
        with GradCAM(model=model, target_layers=target_layer) as cam:
            cam.batch_size = 1 

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_image, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            visualization = show_cam_on_image(test_image, grayscale_cam, use_rgb=True)
    return test_image, visualization, predicted_label, target_answer