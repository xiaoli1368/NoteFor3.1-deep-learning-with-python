# 《Python深度学习》学习笔记

日期：2019/07/09
作者：xiaoli
字数：10k

## 前言

这是我在学习《Python深度学习》中的笔记，这本书的原书是《Deep Learning with Python》，是一本十分通俗易懂的书，因此有必要进行深入的学习。

## 目录

[TOC]

## 第1章 什么是深度学习

> 对于未来或当前的机器学习从业者来说，重要的是能够从噪声中识别出信号，从而在过度炒作的新闻稿中发现改变世界的重大进展。

人工智能、机器学习与深度学习的关系那张图十分重要，见P2。

> 图灵的这个问题引出了一种新的编程范式。在经典的程序设计（即符号主义人工智能的范式）中，人们输入的是规则（即程序）和需要根据这些规则进行处理的数据，系统输出的是答案（见图 1-2）。利用机器学习，人们输入的是数据和从这些数据中预期得到的答案，系统输出的是规则。这些规则随后可应用于新的数据，并使计算机自主生成答案。

上面那段话也很重要

你可以将深度网络看作多级信息蒸馏操作：信息穿过连续的过滤器，其纯度越来越高（即对任务的帮助越来越大）。

人们曾对人工智能极度乐观，随后是失望与怀疑，进而导致资金匮乏。这种循环发生过两次。

> 当前工业界所使用的绝大部分机器学习算法都不是深度学习算法。深度学习不一定总是解决问题的正确工具：有时没有足够的数据，深度学习不适用；有时用其他算法可以更好地解决问题。如果你第一次接触的机器学习就是深度学习，那你可能会发现手中握着一把深度学习“锤子”，而所有机器学习问题看起来都像是“钉子

> 深度学习还让解决问题变得更加简单，因为它将特征工程完全自动化，而这曾经是机器学习工作流程中最关键的一步。

如何了解机器学习算法和工具的现状：
> 要想了解机器学习算法和工具的现状，一个好方法是看一下 Kaggle 上的机器学习竞赛。

>就数据而言，除了过去 20 年里存储硬件的指数级增长（遵循摩尔定律），最大的变革来自于互联网的兴起，它使得收集与分发用于机器学习的超大型数据集变得可行

## 第2章 神经网络的数学基础

### 2.1 初识神经网络

这里进行了第一个小demo，是基于Keras的手写数字分类。

这里使用到的数据集是MNIST数据集。

在调试书上给出的代码前，先熟悉以下基本的库和工具

#### 2.1.1 Mnist数据集

首先是使用Ipython来导入Mnist库并查看其相应的功能：

```python
# 输入
from keras.datasets import mnist
print(help(mnist))
```

```python
# 输出
'''
Help on module keras.datasets.mnist in keras.datasets:

NAME
    keras.datasets.mnist - MNIST handwritten digits dataset.

FUNCTIONS
    load_data(path='mnist.npz')
        Loads the MNIST dataset.
        
        # Arguments
            path: path where to cache the dataset locally
                (relative to ~/.keras/datasets).
        
        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

DATA
    absolute_import = _Feature((2, 5, 0, 'alpha', 1), (3, 0, 0, 'alpha', 0...
    division = _Feature((2, 2, 0, 'alpha', 2), (3, 0, 0, 'alpha', 0), 8192...
    print_function = _Feature((2, 6, 0, 'alpha', 2), (3, 0, 0, 'alpha', 0)...

FILE
    d:\anaconda3\lib\site-packages\keras\datasets\mnist.py
```

从上边可以看到Mnist库是keras中内部集成了的手写数字的数据库，而直接按照格式调用后发现数据的格式为：

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
```

```python
# 输出
'''
(60000, 28, 28)
(60000,)
(10000, 28, 28)
(10000,)
```

可以发现，对于训练集，训练的图片是一个三维的数据立方，是由6万张图片构成，每个图片是28*28的像素格式。而训练集的标签为一维的向量，即6万个类别，其中覆盖0到9。而测试集的数据也是对应的，只不过总共有1万张照片。

以下将训练集中的四副图及其标签画出来，结果如下所示：

```python
import matplotlib.pyplot as plt
import numpy as np

# 载入数据集，并且输出维度大小
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 随机提取四副图片
index = np.random.randint(0, 60000, 4)

# 绘制四副图片
plt.figure(figsize=(10, 10))                         # 设置图片画布大小
plt.subplots_adjust(wspace = 0.1, hspace = 0.3)      # 设置子图间隔大小
for i in range(4):
    plt.subplot(2,2,i+1)
    title = 'label:'+str(train_labels[index[i]])     # 中间的+号表示字符串拼接，真是神奇啊
    plt.title(title, fontsize = 15)                  # 设置图片标题以及字体大小
    plt.xticks(fontsize = 15)                        # 设置x刻度大小
    plt.yticks(fontsize = 15)                        # 设置y刻度大小
    plt.imshow(train_images[index[i],:,:]) 
plt.show()
```

![2.1.1](figure/2.1.1.png)

可以看到训练集中的图像和标签（即图像的标题）是一致的

#### 2.1.2 原始手写数字识别代码

书上给的源代码，仅给出运行结果为：

```python
# 2-1-Mnist.py
from keras.datasets import mnist       
from keras.utils import to_categorical 
from keras import models			  
from keras import layers             

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
 				loss='categorical_crossentropy',
 				metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images  = test_images.reshape((10000, 28 * 28))
test_images  = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
```

```python
Epoch 1/5
60000/60000 [===========================] - 60s 1ms/step - loss: 0.2579 - acc: 0.9258
Epoch 2/5
60000/60000 [===========================] - 61s 1ms/step - loss: 0.1038 - acc: 0.9693
Epoch 3/5
60000/60000 [===========================] - 76s 1ms/step - loss: 0.0695 - acc: 0.9783
Epoch 4/5
60000/60000 [===========================] - 66s 1ms/step - loss: 0.0501 - acc: 0.9855
Epoch 5/5
60000/60000 [===========================] - 65s 1ms/step - loss: 0.0379 - acc: 0.9887
Test  1/1
10000/10000 [===========================] - 19s 2ms/step - test_acc: 0.9774
```

#### 2.1.3 to_categorica 函数

这里需要明白其中各个函数的意义，首先是 to_categorica 函数

```python
print(help(to_categorical))
```

```python
# 输出
'''
Help on function to_categorical in module keras.utils.np_utils:
 
to_categorical(y, num_classes=None, dtype='float32')
    Converts a class vector (integers) to binary class matrix.
    
    E.g. for use with categorical_crossentropy.
    
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

None
# Example
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
```

可以看到 to_categorica 函数的做账用是将一个整型向量，转换为一个二进制类的矩阵，例如将一个1*500的一维向量，但是其中只有6类，转换为一个6\*500的二维矩阵，即原来的每一类使用6位0/1的组合来进行表示。

```python
train_labels
```

```python
# 输出
'''
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
```

```python
train_labels = to_categorical(train_labels)
train_labels
```
```python
# 输出
'''
array([[0., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.]], dtype=float32)
```


```python
train_labels.shape
```

```python
# 输出
'''
(60000, 10)
```

可以看到在本例中，train_labels是大小为60000的一维向量，其中每个元素代表对应训练集中图片的类别，数据类型是unit8，而经过 to_categorica 函数处理后，变为60000\*10 大小的二维矩阵，这是由于原始的数据中共有0-9共10类导致，因此该函数自动统计了共有多少类别，同时每个类别使用了10组0/1的组会来进行表示。

#### 2.1.4 model模型

model是keras中的模型，常用的有两种模型，一种是Sequential 顺序模型，另外一种是[使用函数式 API 的 Model 类模型](https://keras.io/models/model)。这里可以通过以下两种方式来创建模型：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

对Dense而言，input_dim和input_length是等价的。

#### 2.1.5 模型编译

模型编译其实也是配置学习过程的步骤，主要使用complie函数，关键的三个参数为：

- 优化器 optimizer
- 损失函数 loss
- 评估标准 metrics

常见范例如下：

```python
# 多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 均方误差回归问题
model.compile(optimizer='rmsprop',
              loss='mse')

# 自定义评估标准函数
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

#### 2.1.6 模型训练

主要使用fit函数，主要的几个参数为：

- x：训练集样本
- y：训练集标签
- epochs：整数，训练模型迭代轮次
- batch_size：整数，每次梯度更新的样本数，批大小。默认每次更新一批32个数据。

#### 2.1.7 模型评估

主要使用evaluate函数，使用测试模式，返回误差值和评估标准值。

主要是两个参数：测试集样本，以及测试集标签。

#### 2.1.8 修改版代码

以下是自己的修改版：

```python
# 载入必须的库
from keras.datasets import mnist       # 这是数据集
from keras.utils import to_categorical # 这是用于分类的工具
from keras import models               # 模型
from keras import layers               # 层
import matplotlib.pyplot as plt        # 绘图工具

# 提取原始数据，以下是固定的形式，由mnist数据集决定
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28 * 28)) # 改为二维数组
train_images = train_images.astype('float32') / 255   # 归一化
test_images  = test_images.reshape((10000, 28 * 28))  # 改为二维数组
test_images  = test_images.astype('float32') / 255    # 归一化

# 将类向量（整数）转换为二进制类矩阵
train_labels = to_categorical(train_labels)  # 分类编码
test_labels  = to_categorical(test_labels)   # 分类编码

# 留出验证集（这里是新增加的内容）
x_val 			= train_images[:10000] # 验证集：前10000个
partial_x_train = train_images[10000:] # 训练集：10000至60000个
y_val 			= train_labels[:10000] # 验证集：前10000个
partial_y_train = train_labels[10000:] # 训练集：10000至60000个

# 建立网络模型
network = models.Sequential()  		   # 顺序模型
# 第一层：全连接层，512路relu激活函数
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))   
# 第二层：全连接层，10路softmax激活函数
network.add(layers.Dense(10, activation = 'softmax'))							

# 配置训练模型
network.compile(optimizer = 'rmsprop',                  # 优化器，rmsprop
 				loss      = 'categorical_crossentropy', # 代价函数, 交叉熵
 				metrics   =['accuracy'])                # 评估指标，精度

# 进行网络训练
history = network.fit(partial_x_train, partial_y_train, epochs=5, batch_size=128, 						  validation_data=(x_val, y_val))   # 5轮次，每一批128个

# 查看测试集结果
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# 作图
history_dict    = history.history
loss_values     = history_dict['loss']
val_loss_values = history_dict['val_loss 
epochs = range(1, len(loss_values) + 1) # 初始化迭代轮次向量
plt.figure(1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history_dict['acc'] 
val_acc = history_dict['val_acc']
plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

结果如下所示：

![2.1.8a](/figure/2.1.8a.png)

![2.1.8a](/figure/2.1.8b.png)

对于上图的分析：

其中这个例子十分的简单，以至于不到30行的代码便可以完成对于几万幅图像的基于深度学习的分类识别。可以发现，训练集损失函数始终处于下降的趋势，并且训练集精度始终是逐渐提高。这就是梯度下降法的预期效果。

最终测试集精度为 97.37%，比训练集精度98.71%低一些，这是十分正常的，因为我们的神经网络只是学习了训练集的特征，还不能获得很好的泛化误差。可以从上述两幅图观察到，在第4轮迭代后验证集损失开始增大，并且验证集上的精度出现下降。**训练精度和测试精度之间的这种差距是过拟合成的**，一般来说随着神经网络在训练数据上的表现越来越好，模型最终会过拟合，在这种情况下，为了防止过拟合，可以在4轮之后停止训练。而常见的预防过拟合的方法如使用较小的网络，这需要额外的实验来进行测试。

除此之外，对于模型构建中关键参数的选择也是一个值得研究的课题，其实这也属于调参的大范围，如优化器的选择问题。有一种说法是“无论你的问题是什么，rmsprop优化器通常都是足够好的选择”，我认为不能盲从，需要从理论上分析为什么rmsprop具有广泛的适用场景。

关于这个实验还有很多值得思考的地方，需要进一步的测试和验证。



### 2.2 神经网络的数据表示

#### 2.2.1 基本概念

这里引入了一个新的概念，称之为**张量**，英文称之为**tensor**，并且谷歌的TensorFlow的含义就是流动的张量，

> tensorflow的命名来源于本身的运行原理，tensor(张量)意味着N维数组，flow（流）意味着基于数据流图的计算，所以tensorflow字面理解为张量从流图的一端流动到另一端的计算过程。

首先是关于张量的定义：

**0D张量**：也就是标量，这里为什么是0D呢，因为D的意思是Demision，即维度，而一个轴（直线）则表示一个维度，对一个标量而言，其维度是1\*1，也就是一个点，并不能构成一条直线，因此只能是0D。

**1D张量**：也就是向量，维度是1*N，注意只有一个轴，

**2D张量**：也就是二维矩阵，或者说数组，维度是M*N，注意有两个轴



注意张量的几个性质：

- 形状，即size，或者称之为shape，（3， 3， 5）
- 维度，ndim，其实就是shape的size
- 数据类型：dtype，一般都是数值类型的。



张量的三个函数：shape，ndim，dtype

实际生活中的张量

#### 2.2.2 张量切片

张量切片：这个操作很重要

#### 2.2.3 张量运算 

**张量运算：**

> 所有计算机程序最终都可以简化为二进制输入上的一些二进制运算（AND、OR、NOR 等）

深度神经网络学到的所有变换也都可以简化为数值数据张量上的一些张量运算（

rule函数背后对应的数学张量运算。

自己编写的 relu函数



**逐元素的运算**

**广播**

广播运算的自我编写

**张量点积**

对张量点积的理解，以及手写各种点积的原始的py代码

（有时间可以写一写matlab的代码）

**张量变形**



## 第三章 神经网络入门



### 3.1 神经网络基本知识

层：

dense类（密集连接层，全连接层，密集层）

循环层

二维卷积层



模型：指的是，由层构成的网络，有向无环图



损失函数：

优化器：



常见的损失函数的选择准则：

- 二分类：二元交叉熵
- 多分类：分类交叉熵
- 回归问题：均方误差
- 序列学习：联结主义时序分类



### 3.2 Keras简介

首先，Keras是一个深度学习框架。

Keras 基于宽松的 MIT 许可证发布，这意味着可以在商业项目中免费使用它。

Keras 没有选择单个张量库并将 Keras 实现与这个库绑定，而是以模块化的方式处理这个问题（见图 3-3）。因此，几个不同的后端引擎都可以无缝嵌入到 Keras 中。

目前，Keras 有三个后端实现：TensorFlow 后端、Theano 后端和微软认知工具包（CNTK，Microsoft cognitive toolkit）后端。未来 Keras 可能会扩展到支持更多的深度学习引擎。

典型的Keras开发的例子：

- 定义训练数据
- 定义由层组成的网络，或者模型
- 配置学习过程
- 进行迭代训练



两种配置模型的形式，顺序类和函数式。



如何建立深度学习工作站

- Jupyter
- 

### 3.3 电影评论分类：二分类









































































