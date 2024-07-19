import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import multiprocessing
from torchvision.models import MobileNet_V3_Large_Weights
import flwr as fl

# Device configuration
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# Data preprocessing functions
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

# Model definition
def transfer_learning_mobilenetv3(pretrained_path=None):
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(in_features=1280, out_features=4)
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained model from '{pretrained_path}'")
    model.to(device)
    return model

pretrained_model_path = 'mobilenetv3_large_corn_leaf.pth'
model = transfer_learning_mobilenetv3(pretrained_path=pretrained_model_path)

# Flower strategy with model saving callback
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def on_fit_end(self, server_round, results, failures):
        model_path = 'mobilenetv3_large_corn_leaf.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved as '{model_path}'")

# Flower server start
strategy = SaveModelStrategy(model=model)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)