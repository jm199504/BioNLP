# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:28:20 2019

@author: Junming Guo

Email: 2017223045154@stu.scu.edu.cn

Location: Chengdu, 610065 Sichuan Province, P. R. China
"""
import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt

import random

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,LSTM
from keras.optimizers import RMSprop
from keras import backend as K

num_classes = 10
epochs = 20

'''
欧式距离计算（[向量x，向量y]）
'''
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

'''
输出值shape
'''
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

'''
对比损失（真实y，预测y）
'''
def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

'''
生成配对组合（数据x，索引数组=[[1,2,5...],[3,4,6,...]共10个子list，分别代表0-9]）
'''
def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

'''
构建基础网络（输入维度）
'''
def create_base_network(input_shape):
#    input = Input(shape=input_shape)
#    x = Flatten()(input)
#    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.1)(x)
#    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.1)(x)
#    x = Dense(128, activation='relu')(x)
#    print(input_shape)
    input = Input(shape=input_shape)
    lstm1 = LSTM(28, return_sequences=True)(input)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(28, return_sequences=True)(dropout1)
    lstm3 = LSTM(28)(lstm2)
    dropout2 = Dropout(0.2)(lstm3)
    x = Dense(10, activation="softmax")(dropout2)
    return Model(input, x)

'''
计算分类准确度（阈值法）
其中.ravel()类似于flatten()均为扁平化操作，区别在于
a = y.ravel() # 返回数组的视图，视图是数组的引用（虽然a,y地址不同）
b = y.flatten()# 分配了新的内存（a,y地址不同）
修改a会改变y的值；而修改b不会改变y的值；建议使用flatten()
'''
def compute_accuracy(y_true, y_pred):
    print(y_true)
    print(y_pred)
    df = pd.DataFrame()
    df['true'] = list(y_true)
    df['pred'] = list(y_pred)
    df.to_csv("a.csv",index=False)
    pred = y_pred.ravel() < 0.5
    print(np.mean(pred == y_true))
    print('='*50)
    return np.mean(pred == y_true)

'''
计算分类准确度（Kmean修正阈值）
'''
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

'''
绘制训练/验证过程曲线
'''
def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

# 获得训练集 & 测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
#print("input_shape:",input_shape)
# (28,28)

# 生成训练集各类（0-9）的索引
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]

# 生成训练集的正负样本配对组合
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

# 生成测试集各类（0-9）的索引
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]

# 生成测试集的正负样本配对组合
te_pairs, te_y = create_pairs(x_test, digit_indices)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# 共用相同权重的网络
processed_a = base_network(input_a)
processed_b = base_network(input_b)

'''
Lambda用于数据变换，不涉及学习参数
function：映射函数（计算欧氏距离）
output_shape：返回值的shape
'''
distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

model.summary()

# 训练过程（定义优化器；模型编译；模型训练）
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,verbose=2,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# 绘图过程（Loss；Accuracy）
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plot_train_history(history, 'loss', 'val_loss')
plt.subplot(1, 2, 2)
plot_train_history(history, 'accuracy', 'val_accuracy')
plt.show()

# 计算训练集和测试集的准确度
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
