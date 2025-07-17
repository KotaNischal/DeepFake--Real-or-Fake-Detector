import os
import cv2
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from google.colab import drive
from resnet_model import ModifiedResNet

# Mount Google Drive
drive.mount('/content/drive')

# Paths
image_real_path = "/content/drive/MyDrive/Data/raw_images/real"
image_fake_path = "/content/drive/MyDrive/Data/raw_images/fake"
train_dir = "/content/drive/MyDrive/Data/image_dataset/train"
val_dir = "/content/drive/MyDrive/Data/image_dataset/validation"
test_dir = "/content/drive/MyDrive/Data/image_dataset/test"

# Make directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# File copying
def copy_files_parallel(paths, labels, destination_folder):
    def copy_file(path, dest):
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(path, dest)
    with ThreadPoolExecutor() as executor:
        futures = []
        for path, label in zip(paths, labels):
            label_folder = "real" if label == 0 else "fake"
            dest = os.path.join(destination_folder, label_folder, os.path.basename(path))
            futures.append(executor.submit(copy_file, path, dest))
        for future in futures:
            future.result()

# Image collection
real_images = [(os.path.join(image_real_path, img), 0) for img in os.listdir(image_real_path) if img.endswith(('png','jpg','jpeg'))]
fake_images = [(os.path.join(image_fake_path, img), 1) for img in os.listdir(image_fake_path) if img.endswith(('png','jpg','jpeg'))]
all_images = real_images + fake_images
if not all_images: raise ValueError("No images found!")
paths, labels = zip(*all_images)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(paths, labels, test_size=0.3, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)
copy_files_parallel(train_paths, train_labels, train_dir)
copy_files_parallel(val_paths, val_labels, val_dir)
copy_files_parallel(test_paths, test_labels, test_dir)

# Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
train_loader = DataLoader(CustomImageDataset(train_paths, train_labels, transform), batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(CustomImageDataset(val_paths, val_labels, transform), batch_size=32, shuffle=False, num_workers=4)

# Training
model = ModifiedResNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_loss = float('inf')
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_loss, correct, total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {correct/total:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/content/drive/MyDrive/Data/image_classification_model.pth")

train_model(model, train_loader, val_loader, criterion, optimizer)
