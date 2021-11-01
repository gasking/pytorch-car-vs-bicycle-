# 1. 目录介绍
  # 1.1 数据集存放
    bicycle
    car
    phone
# 1.2 models下面存放我们的主干网络

# 2. 运行
 直接运算eval.py文件

# 3.训练自己的模型
  # 1. 将自己的数据集用不同的文件夹存放且文件夹的名称为该要识别的类别
     example:
     我们想识别2个类别 分别为手和头
     于是我们新建2个文件夹
     名命为:
        head
        hand
     然后将对应的图片放入相应的文件夹
   # 2. 在cus.yml里面
     修改 classes=["bicycle",'car','phone']
     --> classes=['head','head']
   
   # 3. 然后运行main.py 即可训练自己的数据集
   #  4. 环境中缺少什么库使用  pip install xxx -i https:mirrors.aliyun.com/pypi/simple
