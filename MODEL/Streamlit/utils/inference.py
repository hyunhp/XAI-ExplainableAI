# Standard library imports
import base64
import io
import json
import os
import tempfile
import warnings
from typing import Tuple

# Third-party library imports for data handling
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

# Machine learning and computer vision libraries
import cv2
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Suppress specific warnings from libraries
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')

def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert the image to RGB format if not already (some images might be in palette mode)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save the resized image to a bytes buffer to avoid writing back to disk
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        
        # Encode the image data to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def Set_pretrained_model(num_classes:int, pretrained_model_path:str) -> Tuple[torch.nn.Module, str]:
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

def caption_inference(target_image, model, model_architecture, device, transform, label_mapping, label_dic):
    '''
    OUTPUT 
        test_image : Converted target image to 224 * 224 array datatype
        visualization : Concated target image with cam result image
        predicted_label : Classification result from model
        target_answer : Answer from target image
        max_activation_coord : Most CAM model focused on the pixel location (row, column)
    '''
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
        with LayerCAM(model=model, target_layers=target_layer) as cam:
            cam.batch_size = 1 

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_image, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            # Get the max gradcam pixel location
            max_activation_coord = np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)

            visualization = show_cam_on_image(test_image, grayscale_cam, use_rgb=True)
            
    return test_image, visualization, predicted_label, target_answer, max_activation_coord


def cam_inference(target_image, model, model_architecture, device, transform, label_mapping):
    '''
    OUTPUT 
        test_image : Converted target image to 224 * 224 array datatype
        visualization : Concated target image with cam result image
        predicted_label : Classification result from model
        target_answer : Answer from target image
        max_activation_coord : Most CAM model focused on the pixel location (row, column)
    '''
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, target_image.name)

    with open(img_path, 'wb') as img_file:
        img_file.write(target_image.getvalue())
    
    # Getting the base64 string
    base64_image = encode_image(img_path)
    
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
        with LayerCAM(model=model, target_layers=target_layer) as cam:
            cam.batch_size = 1 

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        # Get the max LayerCAM pixel location
        max_activation_coord = np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)

        visualization = show_cam_on_image(np.array(test_image), grayscale_cam, use_rgb=True)
        visualization = Image.fromarray(np.uint8(visualization)).convert('RGB')
        
        draw = ImageDraw.Draw(visualization)
        
        # Add a black dot and text using PIL
        draw.ellipse([(max_activation_coord[0]-5, max_activation_coord[1]-5), 
                      (max_activation_coord[0]+5, max_activation_coord[1]+5)], 
                     fill='black', outline='black')
        
        # Define font for the text, if you have a .ttf file use ImageFont.truetype()
        try:
            font = ImageFont.truetype("arial.ttf", 10)  # Adjust path and size
        except IOError:
            font = ImageFont.load_default()

        coord_text = f"Coordinates Point:({max_activation_coord[0]}, {max_activation_coord[1]})"
        # Calculate text size and position
        text_width, text_height = draw.textsize(coord_text, font=font)
        rectangle_back = [(10, 10), (10 + text_width + 6, 10 + text_height + 6)]
        text_x = rectangle_back[0][0] + (rectangle_back[1][0] - rectangle_back[0][0] - text_width) // 2
        text_y = rectangle_back[0][1] + (rectangle_back[1][1] - rectangle_back[0][1] - text_height) // 2

        # Draw semi-transparent rectangle behind text
        draw.rectangle(rectangle_back, fill=(0, 0, 0, 128))  # Black with half transparency
        draw.text((text_x, text_y), coord_text, font=font, fill='white')  # Text on top of rectangle

    return test_image, visualization, predicted_label, max_activation_coord, base64_image


def vlm_inference(input_text:str, base64_image:base64, headers:dict, openaiModel:str='gpt-4o', max_token:int=1024):
    '''
    INFERENCE Visua
    '''
    payload = {
        "model": openaiModel,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text
                        },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ]
                }
            ],
        "max_tokens": max_token
        }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json().get('choices', '')[0].get('message', 'content').get('content', '')
    return response