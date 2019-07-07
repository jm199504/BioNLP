## Siamese LSTM network

Siamese LSTM network（孪生LSTM模型）

 

孪生网络定义：两个Network相同且共享权重。

 

孪生LSTM网络定义：两个LSTM共享权重。



孪生网络结构：

图1



衍生：两个Network不共享权重，表示为Pseudo-siamese network（伪孪生神经网络）。

 

应用：

1. 2010年Hinton在ICML上发表了文章《Rectified Linear      Units Improve Restricted Boltzmann Machines》用于人脸验证。
2. 养乐村同志在NIPS 1993上发表了论文《Signature      Verification using a ‘Siamese’ Time Delay Neural Network》用于美国支票上的签名验证，即验证支票上的签名与银行预留签名是否一致。
3. 计算两个句子或者词汇的语义相似度（较常见）。

 

孪生网络常见损失函数：

1.Contrastive Loss

Hadsell在CVPR 2006大会发表的《Dimensionality       Reduction by Learning an Invariant Mapping》中提及

图2

其中d=||an-bn||2，代表样本特征间的欧氏距离了，y表示为两样本是否匹配（匹配或者相似为1，不匹配为0），margin为超参数（设置的阈值）。

该损失函数表达了对样本的匹配程度，当y=1时（即样本相似）时，损失函数只剩下前半部分（∑yd2），此时如果样本间欧氏距离越大表示其模型性能越差；当y=0是（即样本不相似）时，损失函数只剩下后半部分（∑(1−y)max(margin−d,0)2），其欧氏距离越大表示模型性能越好。

2.Cosine distance

余弦距离，也称为余弦相似度/余弦相似性

图3

其中A，B分别表示两个向量，其Am表示向量A中第m个元素值。

3.Exp Function

图4

其中Y表示真实值标签，f（X）表示预测值标签。

4.Euclidean      distance

欧氏距离，也称欧几里得距离/欧几里得度量

图5

其中公式展示为2维(x,y)的欧氏距离计算公式，可以拓展到N维

拓展：三胞胎网络（Triplet network）《Deep metric learning using Triplet network》

 

第1篇论文：Siamese Recurrent Architectures for Learning Sentence Similarity

 

传统的比较两段文本之间相似性是使用词袋模型TF-IDF模型，存在问题：没有使用上下文信息，词间联系不够密切，泛化能力较弱，对于句子语义理解能力较弱。

 

Siamese Recurrent Architectures 将两个变长语句分别encode成相同长度的向量，再比较两个句子的相似性。

 

Siamese Recurrent Architectures（论文提供图）

图6

输入维度：300；隐藏维度：50

 

实验中效果最好的相似度方程：曼哈顿距离 Manhattan distance

 

曼哈顿距离：d(i,j) = |Xi-Xj|+|Yi-Yj|，其中i和j为2维坐标系中2个点，如i（Xi,Yi）

 

损失函数：均方误差 Squared-error (MSE)

 

第2篇论文：Learning Text Similarity with Siamese Recurrent Networks

 

共享权值的双向LSTM，通过比较句子对之间的相似度信息，将变长的文本映射成固定长度的向量，另外不同之处在于是character-level，可以解决未登录词的问题。





1. 