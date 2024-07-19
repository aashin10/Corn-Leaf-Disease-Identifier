import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import multiprocessing
from torchvision.models import MobileNet_V3_Large_Weights

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def data_preprocess_aug(data_dir, target_size=(224, 224)):
    train_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(root=data_dir, transform=train_transform)
    return train_data


def data_preprocess_non_aug(data_dir, target_size=(224, 224)):
    test_transform = transforms.Compose([
        transforms.Resize(size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(root=data_dir, transform=test_transform)
    return data


# Define data directories
data_dir_train = 'assets/images/Train'
data_dir_test = 'assets/images/Test'
data_dir_val = 'assets/images/Validate'

train_data = data_preprocess_aug(data_dir_train)
test_data = data_preprocess_non_aug(data_dir_test)
val_data = data_preprocess_non_aug(data_dir_val)

# Get the number of CPU cores for parallel data loading
num_cores = multiprocessing.cpu_count()

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)


def transfer_learning_mobilenetv3():
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(in_features=1280, out_features=4)
    model.to(device)
    return model


def train_model(model, train_loader, val_loader, test_loader, device, num_epochs=15,
                accumulation_steps=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()  # Clear any existing gradients

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Scale the loss by the accumulation steps
            loss = loss / accumulation_steps
            loss.backward()

            # Accumulate gradients for a number of steps and then step the optimizer
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # Clear the gradients for the next accumulation

            running_loss += loss.item() * accumulation_steps  # Adjust back the scaled loss for logging
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Perform the scheduler step
        scheduler.step()

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_epoch_acc = correct / total

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        # Testing
        #test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, ')
              #f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    print('Training complete')


# Function to evaluate the model
def evaluate_model(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


model = transfer_learning_mobilenetv3()

num_epochs = 15
#train_model(
#    model, train_loader, val_loader, test_loader, device, num_epochs=num_epochs
#)
#test_loss, test_acc = evaluate_model(model, test_loader, device)
#print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Save the model
#model_path = 'mobilenetv3_large_corn_leaf.pth'
#torch.save(model.state_dict(), model_path)
#print(f"Model saved as '{model_path}'")