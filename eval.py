import  cv2 as cv
import  sys
import torch
from  dataset import  data
import torch.nn.functional as F
import torch.optim as optim#优化器
from torch.autograd import Variable
import random
from  torch.optim.lr_scheduler import StepLR
import  torchvision.transforms as transforms
size = 224
from  models import alexnet,ResNet34
from data_imread import imread
classes=imread()['classes']
def test_val():
    #model=alexnet.AlexNet()
    model=ResNet34.ResNet34()
    model_path=r'2.pth'
    model.load_state_dict(torch.load(model_path),False)
    while True:
     with torch.no_grad():#防止报错
        img_path=input("\033[33;44m [info]  请输入需要分类的图片: \033[0m")
        image=cv.imread(img_path)#变为单通道
        image=cv.resize(image,(size,size))
        tran = transforms.ToTensor()
        img = tran(image)
        img = img.unsqueeze(0)
        predict = model(img)
        index=torch.argmax(predict,dim=1)
        print("预测值:",classes[int(index)])#强制类型转换
        #print(predict)
        #print("得分:","%.5f"%float(-predict[:,int(index)] if predict[:,int(index)]<0 else predict[:,int(index)]))
        cv.imshow("result",image)
        cv.waitKey(0)
test_val()