import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset Class
class CamVidDataset(Dataset):
    def __init__(self, base_dir, split='train'):
        self.base_dir = base_dir
        self.split = split

        # Path Setup
        if split == 'val':
            self.image_dir = os.path.join(base_dir, 'val')
            self.label_dir = os.path.join(base_dir, 'val_labels')
        else:
            self.image_dir = os.path.join(base_dir, split)
            self.label_dir = os.path.join(base_dir, f"{split}_labels")

        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]

        # Load class dictionary
        self.class_dict = pd.read_csv(os.path.join(base_dir, 'class_dict.csv'))

        # Create RGB to class index mapping
        self.color_mapping = {}
        for idx, row in self.class_dict.iterrows():
            rgb = (row['r'], row['g'], row['b'])
            self.color_mapping[rgb] = idx

        # Define transformations
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(360, 480),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(360, 480),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
      # Load image and mask
      img_name = self.images[idx]
      img_path = os.path.join(self.image_dir, img_name)
      mask_name = img_name.replace('.png', '_L.png')
      mask_path = os.path.join(self.label_dir, mask_name)

      image = np.array(Image.open(img_path).convert('RGB'))
      mask = np.array(Image.open(mask_path).convert('RGB'))

      # Convert RGB mask to class indices
      # Change dtype to np.int64 to match with Long tensor type
      label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
      for rgb, idx in self.color_mapping.items():
          label[(mask[:,:,0] == rgb[0]) &
              (mask[:,:,1] == rgb[1]) &
              (mask[:,:,2] == rgb[2])] = idx

      # Apply transformations
      augmented = self.transform(image=image, mask=label)
      image = augmented['image']
      # Explicitly convert mask to Long tensor
      label = torch.as_tensor(augmented['mask'], dtype=torch.long)

      return image, label

# Model Creation
def create_model(num_classes):
    """Create FCN model with ResNet50 backbone"""
    model = fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model

# Training Functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    with tqdm(dataloader) as tqdm_loader:
        for images, targets in tqdm_loader:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.long)  # Ensure Long type

            optimizer.zero_grad()
            outputs = model(images)['out']

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tqdm_loader.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_pixel_acc = 0
    num_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.long)  # Ensure Long type

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            for pred, target in zip(preds, targets):
                intersection = torch.logical_and(pred == target, target == target).sum()
                union = torch.logical_or(pred == target, target == target).sum()
                iou = (intersection.float() / (union.float() + 1e-8)).item()
                total_iou += iou

                correct_pixels = (pred == target).sum().item()
                total_pixels = torch.numel(target)
                total_pixel_acc += correct_pixels / total_pixels

                num_samples += 1

    return total_iou / num_samples, total_pixel_acc / num_samples

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_iou = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_iou, val_pixel_acc = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation IoU: {val_iou:.4f}")
        print(f"Validation Pixel Accuracy: {val_pixel_acc:.4f}")

        # Update learning rate
        scheduler.step(val_iou)

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')

# Visualization Function
def visualize_prediction(model, dataset, index, device):
    # Get a sample image and its true mask
    image, true_mask = dataset[index]

    # Prepare image for model
    input_image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_image)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu()

    # Convert tensors to numpy for visualization
    image = image.permute(1, 2, 0).cpu().numpy()
    # Denormalize image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)

    # Create color-coded masks
    true_mask_colored = np.zeros_like(image)
    pred_mask_colored = np.zeros_like(image)

    # Map indices back to RGB colors
    for rgb, idx in dataset.color_mapping.items():
        true_mask_colored[true_mask == idx] = [rgb[0]/255, rgb[1]/255, rgb[2]/255]
        pred_mask_colored[pred_mask == idx] = [rgb[0]/255, rgb[1]/255, rgb[2]/255]

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_colored)
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_colored)
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

# Main execution
def main():
    # Set device and random seed
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets and dataloaders
    base_dir = '/content/drive/MyDrive/CamVid'
    train_dataset = CamVidDataset(base_dir, split='train')
    val_dataset = CamVidDataset(base_dir, split='val')
    test_dataset = CamVidDataset(base_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Create and train model
    num_classes = len(train_dataset.class_dict)
    model = create_model(num_classes).to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    # Test evaluation
    test_iou, test_pixel_acc = evaluate(model, test_loader, device)
    print(f"\nTest IoU: {test_iou:.4f}")
    print(f"Test Pixel Accuracy: {test_pixel_acc:.4f}")

    # Visualize some predictions
    print("\nVisualizing predictions...")
    for i in range(5):
        print(f"\nTest Image {i+1}")
        visualize_prediction(model, test_dataset, i, device)

if __name__ == "__main__":
    main()
