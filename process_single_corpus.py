'''
# import pickle
# def load_pickle(filename):
#     return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
#修改：不对large进行分析，不需要用到pickle
'''
from collections import Counter
def single_list(arr, target):
    return arr.count(target)
#staqc：把语料中的单候选和多候选分隔开
def data_staqc_prpcessing(filepath,save_single_path,save_mutiple_path):
    # 修改1：函数命名意义不明确：cut_staqc_single_or_mutiply
    with open(filepath,'r')as f:
        total_data= eval(f.read())
        f.close()
    qids = []
    for i in range(0, len(total_data)):
        qids.append(total_data[i][0][0])
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    for i in range(0, len(total_data)):
        if(result[total_data[i][0][0]]==1):
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])
    f = open(save_single_path, "w")
    f.write(str(total_data_single))
    f.close()
    f = open(save_mutiple_path, "w")
    f.write(str(total_data_multiple))
    f.close()



if __name__ == "__main__":
    #将staqc_python中的单候选和多候选分开
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_sigle_save ='../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_python_path,staqc_python_sigle_save,staqc_python_multiple_save)

    #将staqc_sql中的单候选和多候选分开
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_sigle_save = '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_sql_path, staqc_sql_sigle_save, staqc_sql_multiple_save)
'''
#修改：使用循环重构代码，将需要处理的文件路径组成一个包含元组的列表，遍历这个列表，将其中的三个元素分别赋值给 data_path, single_save_path, multiple_save_path，然后调用 data_staqc_prpcessing 函数进行处理
data_paths = [('../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt',
              '../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt',
              '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'),
             ('../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt',
              '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt',
              '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt')]

for data_path, single_save_path, multiple_save_path in data_paths:
    data_staqc_prpcessing(data_path, single_save_path, multiple_save_path)
'''


