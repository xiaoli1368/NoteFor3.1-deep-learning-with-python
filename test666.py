# # 首先测试环境是否配置完善
# import tensorflow
# import keras
# print("hello world!")





# # 其次直接输出图片
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
# import numpy as np

# # 载入数据集，并且输出维度大小
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)

# # 随机提取四副图片
# index = np.random.randint(0, 60000, 4) # 这里不能用 np.random.randint(0, 60000, (1, 4)) 不然会报错
# print(index[0])

# # 绘制四副图片
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(train_images[index[i],:,:])
# plt.show()





# 这里直接训练
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