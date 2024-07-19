import flwr as fl
import torch
from collections import OrderedDict
from centralizedMain import model, train_loader, train_model, evaluate_model, test_loader, val_loader, device
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import multiprocessing

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_model(model, train_loader, val_loader, test_loader, device, num_epochs=1)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_model(model, test_loader, device)
        return float(loss), len(test_loader.dataset), {"accuracy": float(accuracy)}


fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=Client().to_client(),
)