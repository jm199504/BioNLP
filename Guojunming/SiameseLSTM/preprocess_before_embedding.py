# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:50:59 2019

@author: Junming Guo

Email: 2017223045154@stu.scu.edu.cn

Location: Chengdu, 610065 Sichuan Province, P. R. China
"""

import pandas as pd
import numpy as np
import gensim

# load document
train = pd.read_csv("BioNLP_train.csv",delimiter="\t")
#print(train.columns)# ['label', 'dict_id', 'entity', 'name']

# load corpus
embedding_file = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file,binary=True)
embedding_size = 300
max_length = 17

def preprocess_words():
    # preprocess before embedding word vectors
    first_entity_list = list()
    second_entity_list = list()
    for i in train.values:
        first_list = list()
        second_list = list()
        
        # Reading embedding from corpus if exists else random initialize
        for f in str(i[2]).split(' '):
            temp = model[f] if f in model.wv.vocab.keys() else np.random.normal(size=embedding_size)
            first_list.append(temp)
            
        for s in str(i[3]).split(' '):
            temp = model[s] if s in model.wv.vocab.keys() else np.random.normal(size=embedding_size)        
            second_list.append(temp)
            
        # add zero vectors into word vectors
        zero_vector = [0]*embedding_size
        while( len(first_list) < max_length ):
            first_list.append(zero_vector)
            
        # convert into (max_length,embedding_size     
        first_list = np.array(first_list).reshape(-1,embedding_size)
        second_list = np.array(second_list).reshape(-1,embedding_size)
        
        # append into entity_list
        first_entity_list.append(first_list)
        second_entity_list.append(second_list)
        
        print(first_list.shape)
        
    return first_entity_list,second_entity_list

#import tensorflow as tf

#def all_pool(variable_scope, x):
#    with tf.variable_scope(variable_scope + "-all_pool"):
#        if variable_scope.startswith("input"):
#            pool_width = s
#            d = d0
#        else:
#            pool_width = s + w - 1
#            d = di
#        all_ap = tf.layers.average_pooling2d(
#            inputs=x,
#            # (pool_height, pool_width)
#            pool_size=(1, pool_width),
#            strides=1,
#            padding="VALID",
#            name="all_ap"
#        )
#        # [batch, di]
#        all_ap_reshaped = tf.reshape(all_ap, [-1, d])
#
#        return all_ap_reshaped

# 执行：词向量预处理
first_entity_list,second_entity_list = preprocess_words()
print(len(first_entity_list),len(second_entity_list))

# 执行：压缩代码
#x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
#x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")
## 每一个词向量通过all-wp压缩成一维
#LO_0 = all_pool(variable_scope="input-left", x=x1)
#RO_0 = all_pool(variable_scope="input-right", x=x2)



    