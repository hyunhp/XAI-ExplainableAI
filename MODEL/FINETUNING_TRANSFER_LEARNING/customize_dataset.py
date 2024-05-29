from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import torch
import random

    
class CustomDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None, augmented_images=None, augmented_labels=None, metadata=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmented_images = augmented_images if augmented_images is not None else []
        self.augmented_labels = augmented_labels if augmented_labels is not None else []

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = pd.read_csv(label_dir)

        # Create a mapping from string labels to integers
        self.label_mapping = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}
        

    def __len__(self):
        # Return the total number of samples, including both original and augmented
        return len(self.metadata) + len(self.augmented_images)
    
    def __getitem__(self, idx):
        if idx < len(self.metadata):
            # Original data
            img_name = self.metadata.iloc[idx, 0]
            label_str = self.metadata.iloc[idx, 1]
            label = self.label_mapping[label_str]
            img_path = os.path.join(self.root_dir, img_name + '.jpg')
            image = Image.open(img_path).convert('RGB')
        else:
            # Augmented data
            idx -= len(self.metadata)
            image = self.augmented_images[idx]
            label = self.augmented_labels[idx]
            
        if self.transform:
            image = self.transform(image)

        return image, label


def load_finetune_dataset(data_dir:str, label_dir:str, MeanStd:dict, batch_size:int=32, 
                           num_workers:int=0, apply_augment:bool=True, apply_random:bool=True):
    '''
    Load target fintune dataset to fine tune on pre-trained model
    Argument
        - data_dir: The directory containing the dataset.
        - label_dir: The directory containing label data in csv file.
        - batch_size: The batch size for loading the dataset.    
        - meanstd : Computed the HAM10000 mean and standard deviation value  
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)) # ImageNet Trained size
        , transforms.ToTensor()
        , transforms.Normalize(mean=MeanStd['Mean'], std=MeanStd['Std'])
    ])

    dataset = CustomDataset(data_dir, label_dir, transform=transform)

    # Split the dataset into training, validation and test sets
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size  = len(dataset) - train_size - valid_size
        
    # Split train, valid, and test data indices
    train_indices, valid_indices, test_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, valid_size, test_size])
    
    # Extracy indices for each subset to apply 
    train_indices = train_indices.indices
    valid_indices = valid_indices.indices
    test_indices = test_indices.indices
        
    # Create the new CustomDataset instance
    train_dataset = CustomDataset(data_dir, label_dir, transform=transform, 
                                augmented_images=[dataset.augmented_images[i - len(dataset.metadata)] for i in train_indices if i >= len(dataset.metadata)],
                                augmented_labels=[dataset.augmented_labels[i - len(dataset.metadata)] for i in train_indices if i >= len(dataset.metadata)],
                                metadata=dataset.metadata.iloc[train_indices])
    valid_dataset = CustomDataset(data_dir, label_dir, transform=transform, metadata=dataset.metadata.iloc[valid_indices])
    test_dataset = CustomDataset(data_dir, label_dir, transform=transform, metadata=dataset.metadata.iloc[test_indices])

    print(f'After splitting: Training dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}, Test dataset size: {len(test_dataset)}')
    
    if apply_augment:
        # Apply data augmentation only to the training dataset
        augmented_train_images, augmented_train_labels, randomness = apply_augmentation(train_dataset, randomness=apply_random)
        
        # Extend the training dataset with augmented images
        train_dataset.augmented_images = augmented_train_images
        
        train_dataset.augmented_labels = augmented_train_labels
        
    else:
        randomness = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'Final: Training dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}, Test dataset size: {len(test_dataset)}')
    
    # Check if the subsets reference the correct instances
    print("Is train_dataset a subset of the original dataset?", isinstance(train_dataset, CustomDataset))
    print("Is valid_dataset a subset of the original dataset?", isinstance(valid_dataset, CustomDataset))
    print("Is test_dataset a subset of the original dataset?", isinstance(test_dataset, CustomDataset))

    return train_loader, valid_loader, test_loader, apply_augment, randomness

def apply_augmentation(dataset, randomness:bool=True):
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
    ]
    augmented_images = []
    augmented_labels = []

    # Load image and label from dataset object
    for idx in range(len(dataset)):
        image, label = dataset[idx]  # call __getitem__
        
        if randomness:
            # Apply a random augmentation to each image
            augmentation = random.choice(augmentations)
            augmented_images.append(transforms.ToPILImage()(augmentation(image)))
            augmented_labels.append(label)
        else:
            # Apply all augmentations to each image
            augmented_images.extend([transforms.ToPILImage()(augmentation(image)) for augmentation in augmentations])
            augmented_labels.extend([label] * len(augmentations))
    
    return augmented_images, augmented_labels, randomness
