## 评估方法

官方提供链接：<https://github.com/Bibliome/bionlp-st>

数据下载链接：<https://sites.google.com/view/bb-2019/dataset>

Preparation：

1. 安装Java并配置环境（版本>8）
2. 安装Apache Maven并配置环境（版本>=3.3）
3. 下载数据（BioNLP-OST-2019_BB-norm_train/dev.zip）

Build：

1. git clone https://github.com/Bibliome/bionlp-st.git
2. cd bionlp-st
3. mvn clean install

Usage：

1. java -jar bionlp-st-core-0.1.jar -help
2. java -jar bionlp-st-core-0.1.jar -list-tasks
3. java -jar -task TASK -train|-dev -prediction PRED [-alternate] [-detailed] [-force]

where

- ```
  TASK
  ```

  is the short name of the task. Available tasks are:

  - `SeeDev-full`
  - `SeeDev-binary`
  - `BB19-norm`
  - `BB19-rel`
  - `BB19-kb`
  - `BB19-norm+ner`
  - `BB19-rel+ner`
  - `BB19-kb+ner`
  - `BB-cat`
  - `BB-event`
  - `BB-kb`
  - `BB-cat+ner`
  - `BB-event+ner`
  - `BB-kb+ner`

- `PRED` is the location of your predictions (`.a2` files), either a directory of a ZIP archive.

- Specify `-alternate` to display several additional measures.

- Specify `-detailed` to display a document-per-document evaluation, including reference-prediction pairings.

- Specify `-force` to evaluate even if errors where found.

具体实操流程：

1. git clone https://github.com/Bibliome/bionlp-st.git

2. 生成Core-jar包

<img src="https://github.com/jm199504/BioNLP/blob/master/Guojunming/Evaluation/image/1.png">

<img src="https://github.com/jm199504/BioNLP/blob/master/Guojunming/Evaluation/image/2.png" width="500">

3. 将jar包和下载的数据集拷贝到项目目录

4. 执行java -jar bionlp-st-core-0.1.jar -task BB19-norm -dev -prediction 6.BioNLP-OST-2019_BB-norm_dev.zip

5. 执行结果

<img src="https://github.com/jm199504/BioNLP/blob/master/Guojunming/Evaluation/image/3.jpg">

评估指标 Jaccard Index（Java代码）：

<img src="https://github.com/jm199504/BioNLP/blob/master/Guojunming/Evaluation/image/4.png" width="500">

评价指标 Jaccard Index（Wikipedia）：

<img src="https://github.com/jm199504/BioNLP/blob/master/Guojunming/Evaluation/image/5.png">

备注：

1. bionlp-st 文件夹已经放入数据集和jar包，可以直接下载执行命令

2. Python版本的Jaccard Index代码——evaluation.py
