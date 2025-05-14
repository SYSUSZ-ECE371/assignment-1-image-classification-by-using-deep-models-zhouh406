import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import os
import time
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def setup_data(data_dir, batch_size=32):
    """Set up data transformations and dataloaders."""
    # Enhanced data augmentation pipeline
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation, only resize and normalize)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(data_dir, data_transforms)
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_temp = random_split(train_dataset, [train_size, val_size])
    
    # Apply validation transforms to validation set
    val_dataset = datasets.ImageFolder(data_dir, val_transforms)
    _, val_dataset = random_split(val_dataset, [train_size, val_size])
    
    # Optimize dataloader with pin_memory for faster data transfer to GPU
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = train_dataset.dataset.classes
    
    return dataloaders, dataset_sizes, class_names

def setup_model(num_classes, use_pretrained=True):
    """Set up model with transfer learning."""
    # Use a more powerful model - ResNet50 instead of ResNet18
    model = models.resnet50(pretrained=use_pretrained)
    
    # Freeze early layers to prevent overfitting
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
        
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                work_dir='work_dir', num_epochs=25, device=None):
    """Train the model with improved monitoring and early stopping."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # For early stopping
    patience = 7
    no_improvement = 0
    
    # For plotting learning curves
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Create work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)
    
    model = model.to(device)
    
    print(f"Training on {device}")
    print(f"Classes: {len(dataloaders['train'].dataset.dataset.classes)}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Use tqdm for progress bar
            loop = tqdm(dataloaders[phase], desc=phase)
            
            # Iterate over data
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                loop.set_postfix(loss=loss.item())
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Store for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pth'))
                print(f"Saved best model to {work_dir}/best_model.pth")
                no_improvement = 0
            elif phase == 'val':
                no_improvement += 1
        
        # Early stopping
        if no_improvement >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break
                
        print()
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'learning_curves.png'))
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Configuration
    data_dir = 'D:/DeskTop/EX2/flower_dataset'
    work_dir = 'D:/DeskTop/EX2/work_dir'
    batch_size = 32
    num_epochs = 25
    
    # Setup data
    dataloaders, dataset_sizes, class_names = setup_data(data_dir, batch_size)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Setup model
    model = setup_model(num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for different layers
    # Higher learning rate for the new layers, lower for pre-trained layers
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 0.001},
        {'params': list(model.parameters())[:-2], 'lr': 0.0001}
    ], weight_decay=0.01)  # AdamW with weight decay for regularization
    
    # Learning rate scheduler - Cosine Annealing with warm restarts
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    
    # Train the model
    model = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, 
        scheduler, work_dir, num_epochs
    )
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
    }, os.path.join(work_dir, 'final_model.pth'))
    
    print(f"Model saved to {work_dir}/final_model.pth")

if __name__ == "__main__":
    main()