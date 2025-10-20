import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = "data"
MODEL_SAVE_PATH = "waste_classifier.pth"
CLASS_NAMES_PATH = "class_names.json"
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def load_data(data_dir):
    image_datasets = {
        split: datasets.ImageFolder(
            Path(data_dir, split),
            data_transforms[split]
        )
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )
        for split in ['train', 'val', 'test']
    }
    
    dataset_sizes = {split: len(image_datasets[split]) for split in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    train_targets = [s[1] for s in image_datasets['train'].samples]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(class_names)
    
    for split, size in dataset_sizes.items():
        print(f"  {split}: {size} images")
    

    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {class_counts[i]} images")
    
    return dataloaders, dataset_sizes, class_names, class_weights

def create_model(num_classes, class_weights):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    for param in model.features[:5].parameters():
        param.requires_grad = False
    
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    return model, criterion, optimizer, scheduler

def train_one_epoch(model, dataloader, criterion, optimizer, dataset_size):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, dataset_size):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    
    return epoch_loss, epoch_acc

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)
        
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, dataset_sizes['train']
        )
        
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion, dataset_sizes['val']
        )
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'✓ New best model saved. (Val Acc: {val_acc:.4f})')
    
    print(f'\nBest Val Acc: {best_acc:.4f}')
    return history

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to 'confusion_matrix.png'")

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history saved to 'training_history.png'")

def main():
    print("="*60)
    print("WASTE CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    dataloaders, dataset_sizes, class_names, class_weights = load_data(DATA_DIR)
    
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"\nClass names saved to '{CLASS_NAMES_PATH}'")
    
    model, criterion, optimizer, scheduler = create_model(len(class_names), class_weights)
    
    print("\nStarting training")
    history = train_model(
        model, dataloaders, dataset_sizes, 
        criterion, optimizer, scheduler, NUM_EPOCHS
    )
    
    plot_history(history)
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    evaluate_model(model, dataloaders['test'], class_names)
    
    print(f"\n✓ Training complete. Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()