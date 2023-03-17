import numpy as np
import random as rd
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from keys import EccKeys
from trainer import Train


class Client:
    def __init__(self, ID, ecc_keys:EccKeys, contrast, trainer:Train, data_url):
        self.client_id = ID
        self.ecc_keys = ecc_keys
        self.contrast = contrast
        self.trainer = trainer
        self.data_url = data_url

    def save_model(self):
        # 保存模型
        pass
    
    def training_process(self):  # 训练过程
        if (self.trainer.train(self.data_url)):
            self.save_model()


    def test(self):  # 测试模型
        self.trainer.test()
            
