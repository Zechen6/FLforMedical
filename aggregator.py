import torch
import numpy as np
from torch import nn
from torch.autograd import Variable as v
import random as rd

clients_registered = ["1", "2", "3"]  # 可参与聚合的用户列表


class Aggregation():
    def __init__(self, aggregation_file=None):
        self.clients = []
        self.aggregation_file = aggregation_file  # 将代码存入txt文件，输入txt文件名字
        self.possibility = []
        self.grad_pool = []

    def addClient(self, clients):  # 用户注册
        for client in clients:
            if client in clients_registered:
                self.clients.append(client)

    def get_possibility(self):  # 获取权重
        if len(self.possibility) == 0:
            total = 0
            for client in self.clients:
                total += len(client.data_set)

            for client in self.clients:
                self.possibility.append(len(client.data_set)/total)

    def aggregate(self):
        if(self.aggregation_file != None):  # 使用默认的聚合方法
            with torch.no_grad:  # 避免纳入计算图以消耗内存
                if len(self.grad_pool) == self.len(self.clients):
                    input_weight_after_aggregate = v(torch.zeros_like(
                        self.clients[0].model.get_weight(0)))  # 初始化每一层的张量
                    hidden_weight_after_aggregate = v(
                        torch.zeros_like(self.clients[0].model.get_weight(1)))
                    output_weight_after_aggregate = v(
                        torch.zeros_like(self.clients[0].model.get_weight(2)))
                    self.get_possibility()
                    i = 0
                    j = 0
                    for client in self.clients:
                        if rd.random() < 0.3:  # 模拟一些客户端无法参与聚合(FedAvg算法)
                            continue
                        input_weight_after_aggregate += torch.mul(
                            client.model.get_weight(0), self.possibility[j])  # 加权聚合
                        hidden_weight_after_aggregate += torch.mul(
                            client.model.get_weight(1), self.possibility[j])
                        output_weight_after_aggregate += torch.mul(
                            client.model.get_weight(2), self.possibility[j])
                        j += 1
                    self.grad_pool = []  # 清空梯度池，等待下一轮重新算

                    for client in self.clients_pool:
                        client.model.set_weight(
                            0, input_weight_after_aggregate)  # 设置权值
                        client.model.set_weight(
                            1, hidden_weight_after_aggregate)
                        client.model.set_weight(
                            2, output_weight_after_aggregate)
                else:  # 使用用户提供的聚合算法
                    with open(self.aggregation_file, "r") as f:
                        code = f.read()
                    exec(code)  # 执行txt文件中的python代码
