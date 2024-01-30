import os
import re
import time
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from Model import *

def label_to_tensor(label):
    if label==0:
        print('标签：人')
        return torch.tensor([1,0,0,0])
    if label==1:
        print('标签：狗')
        return torch.tensor([0,1, 0, 0])
    if label==2:
        print('标签：猫')
        return  torch.tensor([0, 0, 1, 0])
    if label==3:
        print('标签：其他')
        return  torch.tensor([0,0,0,1])

while True:
    try:
        command=int(input("输入1开始训练模型 训练集放在data/种类/jpg图片 输入2开始预测图片 图片放在程序同目录"))
        break
    except:
        continue

if command==1:
    datasets=torchvision.datasets.ImageFolder(root='data',transform=torchvision.transforms.ToTensor())
    load=DataLoader(datasets,shuffle=True)
    for data in load:
        tensor,label=data
        label_tensor=label_to_tensor(int(label))
        train(tensor,label_tensor)
torch.save(obj=model,f='model.pth')

if command==2:
    for path in os.listdir():
        if re.findall(pattern=".jpg",string=path):
            img=Image.open(path)
            tensor=torchvision.transforms.ToTensor()(img)
            output_tensor=model(tensor)
            print(output_tensor)
            output_tensor=output_tensor.detach()
            output_list=output_tensor.tolist()
            index=output_list.index(max(output_list))
            if index==0:
                print(path)
                print("人")
            if index==1:
                print(path)
                print("狗")
            if index==2:
                print(path)
                print("猫")
            if index==3:
                print(path)
                print("其他")
    