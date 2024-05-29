from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def transfer_learning(
    model_architecture,
    num_epochs,
    train_loader, valid_loader, test_loader,
    apply_augment, augment, randomness
):
    model_class = getattr(models, model_architecture)
    model = model_class(weights=True)
    model.fc = nn.Linear(model.fc.in_features, 7)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # can change sgd for computation power
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f'PYTORCH DEVICE : {device}')

    # Training
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Average training loss for the epoch
        average_loss = total_loss / len(train_loader)
        
        # update the learning rate after every epoch
        scheduler.step()
            
        # Validation check by every step and last one
        step:int = 5
        if (epoch % step == step-1) or (epoch == num_epochs-1):
            model.eval()
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.append(preds)
                    all_labels.append(labels)
                    
                    # clear GPU memory explicitly
                    del inputs, labels, outputs, preds
                    torch.cuda.empty_cache()
            
            # Concatenate predictions and labels before moving them to CPU
            all_preds = torch.cat(all_preds, dim=0)
            all_labels= torch.cat(all_labels,dim=0)
            
            # Move predictions and labels to CPU for further processing
            all_preds = all_preds.cpu().numpy()
            all_labels= all_labels.cpu().numpy()
                    
            # Calculate validation accuracy
            val_accuracy = accuracy_score(all_labels, all_preds) * 100
            tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {average_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%...')
            
            # Clear Memory
            del all_preds, all_labels
            torch.cuda.empty_cache()    

    # Test
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Calculate test accuracy
    test_accuracy = accuracy_score(test_labels, test_preds) * 100
    print(f'Test Accuracy : {test_accuracy:.2f}%...')
        
    # Save the fine-tuned model

    if apply_augment:
        pth_name = f'./fine_tune_model/{model_architecture}_iter_{num_epochs}_aug_{augment}_rand_{randomness}_test_{test_accuracy:.2f}%.pth'
    else:
        pth_name = f'./fine_tune_model/{model_architecture}_iter_{num_epochs}_aug_{augment}_test_{test_accuracy:.2f}%.pth'    

    torch.save(model.state_dict(), pth_name)
        
    print(f'Pretrained model saved.....')