import torch
from  dataset import  data
import torch.nn.functional as F
import torch.optim as optim#优化器
from torch.autograd import Variable
import random
from  torch.optim.lr_scheduler import StepLR
import  torchvision.transforms as transforms
from tqdm import tqdm
from  models import alexnet,ResNet34
import argparse
import data_imread
loader=data_imread.imread()
size =loader['image_size']
classes=loader['classes']
def main():
    def arg():
        arg=argparse.ArgumentParser("训练配置参数")
        arg.add_argument("--iter_max",type=str,default=loader['iter_max'],help="这个是一个训练epoch")
        arg.add_argument("--file_name",type=str,default=loader['filename'],help="训练配置文件")
        arg.add_argument("--ratio",type=float,default=loader['ratio'],help='切分训练集')
        arg.add_argument("--learn",type=float,default=loader['learn'],help="训练时的学习率")
        arg.add_argument("--model",type=str,default=loader['model'],help='训练的模型所在路径')
        arg.add_argument("--savemodel",type=str,default='1.pth',help='训练的模型保存路径')
        t=arg.parse_args()
        return t
    ar=arg()
    # Create dummy input
    data_loader=data(ar.file_name,ar.ratio)
    #设置迭代次数
    max_iters=ar.iter_max
    # Create model
    #net=alexnet.AlexNet()
    net=ResNet34.ResNet34()
    if torch.cuda.is_available():
        net.cuda()
    #学习率
    op=optim.Adam(net.parameters(),lr=ar.learn,weight_decay=1e-4)
    _loss=torch.nn.CrossEntropyLoss()
    count=0
    for epoch in tqdm(range(max_iters)):
        if count<data_loader.__len__():
            t=data_loader.__getitem__(count)
            img,traget=t['train'].unsqueeze(dim=0),t['train_label']
            traget=traget.to(dtype=torch.long)
            img=Variable(img)
            traget=Variable(traget)
            if torch.cuda.is_available():
              img,traget=img.cuda(),traget.cuda()
            out=net(img)
            loss=_loss(out,traget)
            op.zero_grad()
            loss.backward()
            op.step()
            count+=1
            tqdm.set_description("loss=%.6f"%(loss.item()))
        else:
            count=random.randint(0,data_loader.__len__()-1)
    torch.save(net.state_dict(),ar.savemodel)
