import torch
import torchvision

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=torch.nn.Conv2d(kernel_size=3,in_channels=3,out_channels=3)
        self.m1=torch.nn.MaxPool2d(kernel_size=3)
        self.c2 = torch.nn.Conv2d(kernel_size=3, in_channels=3, out_channels=3)
        self.m2 = torch.nn.MaxPool2d(kernel_size=3)
        self.n1=torch.nn.Linear(in_features=3*54*54,out_features=500)
        self.n2=torch.nn.Linear(in_features=500,out_features=400)
        self.n3=torch.nn.Linear(in_features=400,out_features=4)
    def forward(self,tensor):
        tensor=tensor.to('cuda')
        tensor=torchvision.transforms.Resize((500,500))(tensor)
        tensor = self.c1(tensor)
        tensor = self.m1(tensor)
        tensor = self.c2(tensor)
        tensor = self.m2(tensor)
        tensor=torch.flatten(tensor)
        tensor=self.n1(tensor)
        tensor=self.n2(tensor)
        tensor=self.n3(tensor)
        return tensor
try:
    model=torch.load(f='model.pth').to('cuda')
    print('模型从本地载入')
except:
    model=Net().to('cuda')
    print('新创建模型')

loss_fn=torch.nn.MSELoss().to('cuda')
optimezer=torch.optim.ASGD(model.parameters(),lr=0.003)

def train(img_tensor,label_tensor):
    img_tensor=torch.tensor(img_tensor,dtype=torch.float).to('cuda')
    label_tensor = torch.tensor(label_tensor, dtype=torch.float).to('cuda')
    output=model(img_tensor)
    print(f"输出值/期望值是{output}/{label_tensor}")
    loss=loss_fn(output,label_tensor)
    optimezer.zero_grad()
    loss.backward()
    optimezer.step()
