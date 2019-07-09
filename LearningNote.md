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
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace = 0, hspace = 0.5)
    plt.title('label:'+str(train_labels[index[i]])) # 中间的+号表示字符串拼接，真是神奇啊
    plt.imshow(train_images[index[i],:,:])
plt.show()
```



可以看到训练集中的图像和标签（即图像的标题）是一致的

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

以下是自己的修改版：

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





























