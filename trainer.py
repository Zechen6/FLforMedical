import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class CustomDataset(Dataset):  # 规定数据最后一列是标签，数据集必须是csv格式
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        features = torch.tensor(sample[:-1].values, dtype=torch.float32)
        label = torch.tensor(sample[-1], dtype=torch.long)
        return {'features': features, 'label': label}


class Train:
    def __init__(self, model, parameters, optimizer, echos, batch_size, loss_fn):
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer
        self.echos = echos
        self.batch_size = batch_size
        self.loss_fn = loss_fn

    def train(self, data_url):  # 要求输入csv文件的地址，该地址可被read_csv()方法解析
        # 数据集的转化
        training_data = CustomDataset(data_url)  # 仅含训练数据，不考虑测试
        # Create data loaders.
        train_dataloader = DataLoader(
            training_data, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for (X, y) in train_dataloader:
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return True
    
    def test(self, data_url):
        # 数据集的转化
        testing_data = CustomDataset(data_url)  # 仅含训练数据，不考虑测试
        # Create data loaders.
        train_dataloader = DataLoader(
            testing_data, batch_size=self.batch_size, shuffle=True)
        sum_loss = 0
        self.model.train()
        for (X, y) in train_dataloader:
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            sum_loss += loss.item()
        return sum_loss