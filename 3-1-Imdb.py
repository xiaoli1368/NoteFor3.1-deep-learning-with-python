from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results # 这个地方，刚才缩进不正确，导致结果差的太大了

    
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')

x_val           = x_train[:10000]
partial_x_train = x_train[10000:]
y_val           = y_train[:10000]
partial_y_train = y_train[10000:]

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1 , activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss      = 'binary_crossentropy',
              metrics   = ['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))

# 结果绘图
history_dict    = history.history
loss_values     = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc             = history_dict['acc']
val_acc         = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(10, 10))                         # 设置图片画布大小
plt.subplots_adjust(wspace = 0.1, hspace = 0.3)      # 设置子图间隔大小
plt.subplot(2, 1, 1)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss', fontsize = 14)
plt.xlabel('Epochs', fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.xticks(fontsize = 14)                        # 设置x刻度大小
plt.yticks(fontsize = 14)                        # 设置y刻度大小
plt.legend(fontsize = 14)

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy', fontsize = 14)
plt.xlabel('Epochs', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.xticks(fontsize = 14)                        # 设置x刻度大小
plt.yticks(fontsize = 14)                        # 设置y刻度大小
plt.legend(fontsize = 14)

plt.show()