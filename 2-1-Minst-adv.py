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

# 留出验证集
x_val = train_images[:10000]
partial_x_train = train_images[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 建立网络模型
network = models.Sequential()  												# 顺序模型
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))   # 第一层：全连接层，512路relu激活函数
network.add(layers.Dense(10, activation = 'softmax'))							# 第二层：全连接层，10路softmax激活函数

# 配置训练模型
network.compile(optimizer = 'rmsprop',                  # 优化器，rmsprop
 				loss      = 'categorical_crossentropy', # 代价函数, 交叉熵
 				metrics   =['accuracy'])                # 评估指标，精度

# 进行网络训练
history = network.fit(partial_x_train, partial_y_train, epochs=5, batch_size=128, 						  validation_data=(x_val, y_val)) # 5轮次，128批大小

# 查看测试集结果并作图
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
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