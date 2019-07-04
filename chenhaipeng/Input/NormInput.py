import pandas as pd
import os
from tqdm import tqdm

def SolveData(a1, a2):
    # 删除起点终点信息只保留名词类别
    df = a1['category+start+end'].str.split(' ', expand=True)
    a1 = a1.drop('category+start+end', axis=1)
    a1.insert(1, 'category', df[0])
    # 删除a2的ID只保留词典类别
    df = a2['a2_id+dic'].str.split('\t', expand=True)
    a2 = a2.drop('a2_id+dic', axis=1)
    a2.insert(0, 'dic', df[1])
    # a2文件中a1 ID前还有单词Annotation，去除
    df = a2['a1_id'].str.split(':', expand=True)
    a2 = a2.drop('a1_id', axis=1)
    a2.insert(1, 'a1_id', df[1])
    # a2文件中词典编号前还有Referent，以及有可能存在OBT(当词典是OntoBiotope时)，去除
    df = a2['obt'].str.split(':', expand=True)
    a2 = a2.drop('obt', axis=1)
    if df.shape[1] == 3:
        df.loc[df[2].isnull(), 2] = df[df[2].isnull()][1]
        a2.insert(2, 'obt', df[2])
    # 如果文章不包含OntoBiotope词典单词则直接补齐插入
    else:
        a2.insert(2, 'obt', df[1])
    # 通过a2文件中关于a1的ID进行两表合并
    result = pd.merge(a1, a2, how='outer', on=['a1_id'])
    result = result.drop('a1_id', axis=1)
    return result

# 读取每一个编号下的a1,a2文件，进行合并处理后输出
def ExtractData(path, file_name):
    # 编号带F的文件不包含前两行的文章信息，因此读取时无需去除前两行信息
    if 'F' in ' '.join(file_name[0]):
        str = ' '.join(file_name[0]).replace(' ', '')
        a1 = pd.read_csv(path + '/' + str, header=None, sep='\t')
        # 类别与文字起点和终点用空格分隔，稍后再单独处理
        a1.columns = ['a1_id', 'category+start+end', 'word']
        str = ' '.join(file_name[1]).replace(' ', '')
        a2 = pd.read_csv(path + '/' + str, header=None, sep=' ')
        # 对应于a1文件的标签与词典名字用tab分隔，稍后单独处理
        a2.columns = ['a2_id+dic', 'a1_id', 'obt']
        result = SolveData(a1, a2)
    else:
        str = ' '.join(file_name[0]).replace(' ', '')
        # 跳过前两行文章信息
        a1 = pd.read_csv(path + '/' + str, skiprows=2, header=None, sep='\t')
        # 类别与文字起点和终点用空格分隔，稍后再单独处理
        a1.columns = ['a1_id', 'category+start+end', 'word']
        str = ' '.join(file_name[1]).replace(' ', '')
        a2 = pd.read_csv(path + '/' + str, header=None, sep=' ')
        # 对应于a1文件的标签与词典名字用tab分隔，稍后单独处理
        a2.columns = ['a2_id+dic', 'a1_id', 'obt']
        result = SolveData(a1, a2)
    return result

# 批量读取文件，记录每一个编号下的a1、a2文件传递给ExtraData进行处理
def TraverFile(file_path):
    g = os.walk(file_path)
    for path, dir_list, f_list in g:
        T = 0
        out = pd.DataFrame(columns=('category', 'word', 'dic', 'obt'))
        pair_file = []
        # 以此读取文件，以同个编号下的三个文件为文件，结合同编号下的a1，a2文件进行数据提取
        file_list = tqdm(f_list)
        for file_name in file_list:
            T += 1
            # 保存a1文件路径
            if T == 1:
                pair_file.append(list(file_name))
            # 加入a2文件路径
            if T == 2:
                pair_file.append(list(file_name))
            # 忽略txt文件并对a1,a2文件进行数据提取
            if T == 3:
                temp = ExtractData(path, pair_file)
                # 提取后的数据加入到待输出的out中
                out = out.append(temp)
                T = 0
                pair_file.clear()
    return out

# 主函数，判断需要的输出数据文件，数据标准化后输出文件为NormInput.csv，表格列名包含category,word,dic,obt
if __name__=="__main__":
    str = input("选择读取文件夹，输入train、dev或者both\n")
    if str == "train":
        train = TraverFile(r"BioNLP-OST-2019_BB-norm_train")
        train.to_csv('./NormInput.csv', index=False, header=True)
    elif str == "dev":
        dev = TraverFile(r"BioNLP-OST-2019_BB-norm_dev")
        dev.to_csv('./NormInput.csv', index=False, header=True)
    elif str == "both":
        train = TraverFile(r"BioNLP-OST-2019_BB-norm_train")
        dev = TraverFile(r"BioNLP-OST-2019_BB-norm_dev")
        pd.concat((train, dev), axis=0).to_csv('./NormInput.csv', index=False, header=True)
    else:
        print('输入错误！')


