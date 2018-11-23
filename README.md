# NumPy-SRCNN
Super-Resolution CNN using numpy

## 1.运行环境
> python 3.x
> OpenCV-Python 3.4.2

## 2.运行方式
> python numpy_srcnn.py image_name learning_rate resize_time epoch
> 执行 python numpy_srcnn.py 默认参数为
>> image_name='./image0.jpg'
>> alpha=3e-11
>> resize_time=2
>> epoch=200

## 3.SR重建网络

#### 用八个中间卷积层，每层的kernel都是3*3，步长为1，out channel为64，再加一个output卷积层，out channel为3，得到重建后的图片。每一个中间卷积层使用Relu激活。因为网络中没有池化层和全连接层，所以输入图片的shape和网络输出矩阵的shape是相同的，可以用均方差来优化网络参数。

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/NumPy-SRCNN/blob/master/imgs/Fig1_1.jpg"/></div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/NumPy-SRCNN/blob/master/imgs/Fig1_2.jpg"/></div>

#### <div align=center>Fig1.训练集生成过程</div>

## 4.运行结果

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/NumPy-SRCNN/blob/master/imgs/loss1.png"/></div>

#### <div align=center>Fig2. loss-epoch</div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/NumPy-SRCNN/blob/master/image0.jpg"/></div>

#### <div align=center>Fig3. 32*32的原图</div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/NumPy-SRCNN/blob/master/output_8_conv/image0/33.jpg"/></div>

#### <div align=center>Fig4. 33*33的模型输出图</div>

<div align=right>模型拟合效果不佳，超清重建做成了高斯模糊，原因待分析...</div>
