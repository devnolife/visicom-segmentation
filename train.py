"""
Training Script untuk Model Segmentasi Banjir
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet, CombinedLoss, get_model
from utils.data_loader import FloodSegmentationDataset, get_train_transform, get_val_transform
from utils.visualization import calculate_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Training satu epoch"""
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validasi satu epoch"""
    model.eval()
    epoch_loss = 0
    all_ious = []
    all_dices = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            epoch_loss += loss.item()
            
            # Calculate metrics
            preds = torch.softmax(outputs, dim=1)[:, 1, :, :]  # Probabilitas kelas positif
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for pred, mask in zip(preds_np, masks_np):
                metrics = calculate_metrics(pred, mask)
                all_ious.append(metrics['iou'])
                all_dices.append(metrics['dice'])
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / len(dataloader)
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)
    
    return avg_loss, avg_iou, avg_dice


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=50,
    checkpoint_dir='checkpoints',
    early_stopping_patience=10
):
    """
    Training loop utama
    
    Args:
        model: Model untuk training
        train_loader: DataLoader untuk training
        val_loader: DataLoader untuk validasi
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (cuda/cpu)
        num_epochs: Jumlah epoch
        checkpoint_dir: Direktori untuk menyimpan checkpoint
        early_stopping_patience: Patience untuk early stopping
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_iou = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'lr': []
    }
    
    print(f"Training dimulai pada device: {device}")
    print(f"Total epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_iou, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, checkpoint_path)
            print(f"✓ Best model saved! IoU: {best_iou:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save last model
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
        }, last_checkpoint_path)
    
    # Plot training history
    plot_history(history, os.path.join(checkpoint_dir, 'training_history.png'))
    
    print("\n" + "="*50)
    print("Training selesai!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    return history


def plot_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(history['val_dice'], label='Val Dice', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='Learning Rate', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Flood Segmentation Model')
    parser.add_argument('--image_dir', type=str, default='dataset/images', help='Image directory')
    parser.add_argument('--mask_dir', type=str, default='dataset/masks', help='Mask directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader = DataLoader(
        FloodSegmentationDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=get_train_transform(args.image_size),
            is_train=True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        FloodSegmentationDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=get_val_transform(args.image_size),
            is_train=True
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    print(f"Creating {args.model} model...")
    model = get_model(args.model, n_channels=3, n_classes=2)
    model = model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    main()
