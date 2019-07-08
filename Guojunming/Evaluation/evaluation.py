import pandas as pd
import os

'''
评估公式：
prediction:预测数据集Entity:OBT_list
real:真实数据集Entity:OBT_list
method:评估方法（默认Jaccard）
'''
def evaluation_formula(prediction,real,method='jaccard'):
	if method == "jaccard":
		inter,union = 0,len(real)
		for p in prediction:
			if p in real:
				inter+=1
			else:
				union+=1
		return inter / union

'''
文件路径指定
file_real_path：真实数据集Entity：OBT路径
file_pred_path：预测数据集Entity：OBT路径
entity_list_path：训练文本的Entity名称
dict_OBT_list_path：标准字典（OBT）
dict_TAX1_trim_list_path：标准字典（非OBT）
'''
file_real_path = "example_data\\train\\real\\"
file_pred_path = "example_data\\train\\pred\\"
entity_list_path = "example_data\\entity_list_BioNLP-OST-2019_BB-norm_train.tsv"

dict_OBT_list_path = "example_data\\OBT.txt"
dict_TAX1_trim_list_path = "example_data\\TAX1_trim.txt"
file_real_list = [file_real_path+_ for _ in os.listdir(file_real_path)]
file_pred_list = [file_pred_path+_ for _ in os.listdir(file_pred_path)]

'''
读取文件列表：
file_list：数据集的a文件列表
输出:实体字典（Entity_id:OBT_list），类型字典（Entity_id:Dict_ype）
'''
def read_files(file_list):
    file_OBT_dict = dict()
    file_type_dict = dict()
    for file in file_list:
        f_out = pd.read_csv(file, sep='\t|\s', header=None)
        f_out.columns=['standard_id', 'dict_type', 'entity_id', 'dict_id']
        f_out.entity_id = f_out.entity_id.str.lstrip('Annotation:')
        f_out.dict_id = f_out.dict_id.str.lstrip('Referent:')
        entity_dict = dict()
        type_dict = dict()
        for i in f_out.values:
            if i[2] in entity_dict.keys():
                entity_dict[i[2]].append(i[3])
            else:
                entity_dict[i[2]] = [i[3]]
            if i[2] in type_dict.keys():
                type_dict[i[2]].append(i[1])
            else:
                type_dict[i[2]] = [i[1]]
        file_OBT_dict[file.split("\\")[-1]] = entity_dict
        file_type_dict[file.split("\\")[-1]] = type_dict
    return file_OBT_dict,file_type_dict
        
real_dict,real_type_dict = read_files(file_real_list)   
pred_dict,pred_type_dict = read_files(file_pred_list)   


'''
评估函数
pred_dict:预测实体字典
real_dict:真实实体字典
pred_type_dict:实体对应类型字典
entity_list_path:实体对应entity
dict_OBT_list_path:OBT字典(标准)
dict_TAX1_trim_list_path:非OBT字典(标准)
'''
def evaluation(pred_dict,real_dict,pred_type_dict,entity_list_path,dict_OBT_list_path,dict_TAX1_trim_list_path):
    entity_name_df = pd.read_csv(entity_list_path,delimiter="\t")
    standard_name_BOT = pd.read_csv(dict_OBT_list_path,delimiter="\t")
    standard_name_TAX = pd.read_csv(dict_TAX1_trim_list_path,delimiter="\t")
    res = pd.DataFrame(columns=('entity_id', 'pred','real','score'))
    for p in pred_dict.items():
        text_id = p[0].split('.')[0][8:]
        pred_item = pred_dict[p[0]]
        real_item = real_dict[p[0]]
        pred_type = pred_type_dict[p[0]]
        for item in pred_item.items():
            entity_id = item[0]
            pred_list = pred_item[item[0]]
            real_list = real_item[item[0]]
            dict_type = pred_type[item[0]]
            score = evaluation_formula(pred_list,real_list)
            input_entity_name_list = list()
            pred_entity_name_list = list()
            real_entity_name_list = list()
            for i in pred_list:
                # 训练文本输出实体名称
                for j in entity_name_df.values:
                    if j[0] == text_id and j[1] == entity_id:
                        input_entity_name_list.append(j[2])
                # 标准字典输出实体名称
                if i[0] == 'O':
                    for m in standard_name_BOT.values:
                        if i == m[0].split('|')[0]:
                            real_entity_name_list.append(m[0].split('|')[1])
                else:
                    for m in standard_name_TAX.values:
                        if i == m[0].split('|')[0]:
                            real_entity_name_list.append(m[0].split('|')[1])
                            
                # 标准字典输出实体名称
                if i[0] == 'O':
                    for m in standard_name_BOT.values:
                        if i == m[0].split('|')[0]:
                            pred_entity_name_list.append(m[0].split('|')[1])
                else:
                    for m in standard_name_TAX.values:
                        if i == m[0].split('|')[0]:
                            pred_entity_name_list.append(m[0].split('|')[1])
                            
            res = res.append([{'text_id':text_id,'dict_type':dict_type[0],
                               'entity_id':entity_id,'pred_entity_name':pred_entity_name_list,
                               'input_entity_name':input_entity_name_list,
                               'pred':pred_list,'real':real_list,
                               'real_entity_name':real_entity_name_list,'score':score}], ignore_index=True)
    
    res.to_csv('result.csv',index=None)
#    return res

'''
统计所有文章的分数均值
'''
def calc_precision(result_path):
    result = pd.read_csv(result_path)
    return result['score'].mean()
'''
运行主函数
将评估结果输出到当前目录\\result.csv
'''
if __name__=='__main__':
    evaluation(pred_dict,real_dict,pred_type_dict,entity_list_path,dict_OBT_list_path,dict_TAX1_trim_list_path)
    result_path = "result.csv"
    print(calc_precision(result_path))
 
    
    
    
    
    
    
    
    
    
