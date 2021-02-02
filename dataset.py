import  os,sys,math,random,shutil
from torch.utils.data import  DataLoader,Dataset
from  PIL import  Image
import numpy as np
import  torch
import  torchvision.transforms as transforms
import data_imread
ag=data_imread.imread()
size=ag['image_size']
classes=ag['classes']
f=open(ag['filename'],'a+')
def dataset():
    path=os.getcwd()
    for ind,value in enumerate(classes):
        try:
            if os.path.exists(os.path.join(path,value)):
                for idx in  os.listdir(os.path.join(path,value)):
                    f.write(
                    os.path.join(os.path.join(path,value),idx)+' '
                    +str(ind)+'\n'
                    )
        except  Exception as e:
            raise  OSError("文件路径不存在")
class data(Dataset):
    def __init__(self,filename=None,ratio=None,train=True,val=True):
        super(data,self).__init__()
        if not os.path.exists(filename):
            dataset()
        self.filename=filename
        self.ratio=ratio
        self.train_flag,self.val_flag=train,val
        self.transform = transforms.Compose(
                 [transforms.ToTensor()])
        self.img_label()
    def  get_train_val(self,file,ratio):
        random.seed()
        ta=[i.split('\n')[0] for i in open(file,'r+').readlines()]
        random.shuffle(ta)
        train=ta[0:math.ceil(len(ta)*ratio)]
        val=ta[math.ceil(len(ta)*ratio):]
        return train,val#这个只是返回图片路径
    def img_label(self):
        train,val=self.get_train_val(self.filename,ratio=self.ratio)
        self.train,self.train_label=self.iters(train)
        self.val,self.val_label=self.iters(val)
    def iters(self,num):
       types=[]
       types_label=[]
       for i in num: 
        p,t=self.split(paths=i)
        types.append(p)
        types_label.append(t)
       return types,types_label
    def loader_image(self,img_path):
        img=Image.open(img_path).resize((size,size),Image.ANTIALIAS)
        return img.convert("RGB")
    def split(self,paths):
        img_path,id=paths.split(' ')
        label=self.num_to_hot(int(id))
        return img_path,label
    def num_to_hot(self,nums):
        _a=np.array([1])
        _a[0]=nums
        t=torch.from_numpy(_a)
        return t
    def __getitem__(self,ind):
        assert len(self.train)!=0,"列表中没有元素"
        return {"train":self.transform(self.loader_image(self.train[ind])),
                "train_label":self.train_label[ind]
                }
    def __len__(self):
        return len(self.train)




