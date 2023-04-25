import torch
import pandas as pd
from tabulate import tabulate

# 定义两个神经网络
net1 = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 40),
    torch.nn.ReLU(),
    torch.nn.Linear(40, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 60),
    torch.nn.ReLU()
)

net2 = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 30),
    torch.nn.ReLU()
)

# 比较两个神经网络
diff1 = []
diff2 = []
for i, (layer1, layer2) in enumerate(zip(net1, net2)):
    if type(layer1) != type(layer2):
        if type(layer1) < type(layer2):
            diff1.append((i, layer1))
        else:
            diff2.append((i, layer2))
    elif isinstance(layer1, torch.nn.Linear) and layer1.weight.shape != layer2.weight.shape:
        if layer1.weight.shape < layer2.weight.shape:
            diff1.append((i, layer1))
        else:
            diff2.append((i, layer2))

# 输出比较结果
table = []
for i, (layer1, layer2) in enumerate(zip(net1, net2)):
    if i in [x[0] for x in diff1]:
        table.append([i, str(diff1[[x[0] for x in diff1].index(i)][1]), ''])
    elif i in [x[0] for x in diff2]:
        table.append([i, '', str(diff2[[x[0] for x in diff2].index(i)][1])])
    else:
        table.append([i, '', ''])
print(tabulate(table, headers=['layer', 'net1_diff', 'net2_diff']))