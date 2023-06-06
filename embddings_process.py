'''
从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
'''
#修改：未使用到以上三个包，建议删除
'''
import numpy as np
import pickle
from gensim.models import KeyedVectors

#词向量文件保存成bin文件
def trans_bin(path1,path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)
    '''n
    读取用一下代码
    model = KeyedVectors.load(embed_path, mmap='r')
    '''

#构建新的词典 和词向量矩阵
def get_new_dict(type_vec_path,type_word_path,final_vec_path,final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(type_word_path,'r')as f:
        total_word= eval(f.read())
        f.close()

    # 输出词向量
    word_dict = ['PAD','SOS','EOS','UNK']#其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID

    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding,sos_embediing,eos_embediing,unk_embediing]
    print(len(total_word))
    for word in total_word:
        try:
            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            print(word)
            fail_word.append(word)
    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    '''
    修改：将失败的词向量忽略，因为这些词在之后的训练中所占比例很小，随机初始化词向量即可。
         将词向量矩阵保存为 numpy 格式，这样可以避免每次使用时都要重新载入。
         将打开文件的操作和关闭文件的操作合并为一步，避免忘记关闭文件。

def get_new_dict(type_vec_path,type_word_path,final_vec_path,final_word_path):
    # 加载词向量
    model = KeyedVectors.load(type_vec_path, mmap='r')
    # 加载词标签
    with open(type_word_path,'r') as f:
        total_word = eval(f.read())
        f.close()
    # 构建新词典和词向量矩阵
    word_dict = ['<PAD>','<SOS>','<EOS>','<UNK>'] # 填充、开始、结束、未知
    word_vectors = [np.zeros((1, 300)).squeeze() for i in range(4)]
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            continue
    # 保存新词典和词向量矩阵
    with open(final_word_path, 'w') as f:
        f.write(str(word_dict))
        f.close()
    np.save(final_vec_path, np.array(word_vectors))
    '''

    #判断词向量是否正确
    '''
    couunt = 0
    #修改：单词拼写错误，影响阅读
    #count=0
    for i in range(4,len(word_dict)):
        if word_vectors[i].all() == model.wv[word_dict[i]].all():
            continue
        else:
            couunt +=1
            #修改：count+=1
    print(couunt)
    #修改：print(count)
    '''

    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    v = pickle.load(open(final_vec_path, 'rb'), encoding='iso-8859-1')
    with open(final_word_path, 'rb') as f:
        word_dict = pickle.load(f)
    '''
    couunt = 0
    #修改：count再之前已经定义并执行过，不知道在此重新定义的意义，删掉
    '''
    print("完成")



#得到词在词典中的位置
def get_index(type,text,word_dict):
    location = []
    if type == 'code':
        location.append(1)
        len_c = len(text)
        if len_c+1 <350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(0, len_c):
                    if word_dict.get(text[i]) != None:
                        index = word_dict.get(text[i])
                        location.append(index)
                    else:
                        index = word_dict.get('UNK')
                        location.append(index)

                location.append(2)
        else:
            for i in range(0, 348):
                if word_dict.get(text[i]) != None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(0, len(text)):
                if word_dict.get(text[i]) != None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)

    return location
'''
#修改：
#将一个函数分解为两个，使人在函数功能上更容易理解；
#在 code 类型的情况下，避免 for 循环进行显式索引，而是使用列表表达式进行迭代，使代码更具可读性；
#在找不到词向量的情况下，直接将其位置赋值为 0，在后续编码过程中，将忽略这些词；
#将一些常量提取为变量，以允许在代码中进行进一步使用。

# 获取词的位置
def get_word_location(text, word_dict):
    word_location = []
    for word in text:
        if word_dict.get(word) is not None:
            index = word_dict.get(word)
            word_location.append(index)
        else:
            word_location.append(0)
    return word_location

# 得到词在词典中的位置
def get_index(type, text, word_dict):
    location = []
    if type == 'code':
        location.append(1)
        len_c = len(text)
        if len_c + 1 < 350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                word_location = get_word_location(text, word_dict)
                location.extend(word_location)
                location.append(2)
        else:
            word_location = get_word_location(text[:348], word_dict)
            location.extend(word_location)
            location.append(2)
    else:
        if len(text) == 0 or text[0] == '-10000':
            location.append(0)
        else:
            word_location = get_word_location(text, word_dict)
            location.extend(word_location)

    return location
'''

#将训练、测试、验证语料序列化
#查询：25 上下文：100 代码：350
def Serialization(word_dict_path,type_path,final_type_path):

    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path,'r')as f:
        corpus= eval(f.read())
        f.close()

    total_data = []


    for i in range(0, len(corpus)):
        qid = corpus[i][0]

        Si_word_list = get_index('text',corpus[i][1][0],word_dict)
        # Si+1
        Si1_word_list = get_index('text',corpus[i][1][1],word_dict)

        # code
        #tokenized_code = get_index('code',corpus[i][2][0],word_dict) #训练语料
        tokenized_code = get_index('code', corpus[i][2][0], word_dict) #staqc
        # query
        query_word_list = get_index('text',corpus[i][3],word_dict)
        #block_length = corpus[i][4]
        #label = corpus[i][5]
        block_length = 4
        label = 0
        if(len(Si_word_list)>100):
            Si_word_list = Si_word_list[:100]
        else:
            for k in range(0, 100 - len(Si_word_list)):
                Si_word_list.append(0)

        if (len(Si1_word_list) > 100):
            Si1_word_list = Si1_word_list[:100]
        else:
            for k in range(0, 100 - len(Si1_word_list)):
                Si1_word_list.append(0)

        if (len(tokenized_code) < 350):
            for k in range(0, 350 - len(tokenized_code)):
                tokenized_code.append(0)
        else:
            tokenized_code = tokenized_code[:350]

        if (len(query_word_list) > 25):
            query_word_list = query_word_list[:25]
        else:
            for k in range(0, 25 - len(query_word_list)):
                query_word_list.append(0)

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)
    '''
    #修改：上述代码中if...else语句过多，修改代码减少if条件语句的使用，为重复的代码创建一个单独的函数。
    def pad_list(data_list, list_len, pad_value=0):
    if len(data_list) > list_len:
        return data_list[:list_len]
    else:
        for k in range(0, list_len - len(data_list)):
            data_list.append(pad_value)
        return data_list
    
    
    WORD_LIST_MAX_LENGTH = 100
    CODE_LIST_MAX_LENGTH = 350
    QUERY_LIST_MAX_LENGTH = 25

    for i in range(0, len(corpus)):
    qid = corpus[i][0]

    Si_word_list = pad_list(get_index('text', corpus[i][1][0], word_dict), WORD_LIST_MAX_LENGTH)
    Si1_word_list = pad_list(get_index('text', corpus[i][1][1], word_dict), WORD_LIST_MAX_LENGTH)
    tokenized_code = pad_list(get_index('code', corpus[i][2][0], word_dict), CODE_LIST_MAX_LENGTH)
    query_word_list = pad_list(get_index('text', corpus[i][3], word_dict), QUERY_LIST_MAX_LENGTH)
    block_length = 4
    label = 0

    one_data = [qid, [Si_word_list + Si1_word_list], [tokenized_code], query_word_list, block_length, label]
    total_data.append(one_data)
    '''
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


def get_new_dict_append(type_vec_path,previous_dict,previous_vec,append_word_path,final_vec_path,final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件

    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    with open(append_word_path,'r')as f:
        append_word= eval(f.read())
        f.close()

    # 输出词向量

    print(type(pre_word_vec))
    word_dict = list(pre_word_dict.keys()) #'#其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word = []
    print(len(append_word))
    rng = np.random.RandomState(None)
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    h = []

    for word in append_word:
        try:

            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            fail_word.append(word)
    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])



    '''
    #判断词向量是否正确
    print("----------------------------")
    count = 0

    import operator
    for i in range(159035,len(word_dict)):
        if operator.eq(word_vectors[i].tolist(), model.wv[word_dict[i]].tolist()) == True:
            continue
        else:
            count +=1

    print(count)
    '''


    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))

    '''
    #修改：map(reversed, enumerate(word_dict))的类型可能不符合规范，但不知如何修改
    '''
    
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)


    print("完成")

import time
'''
#修改：不明所以地导入未使用的time，删掉
'''

#-------------------------参数配置----------------------------------
#python 词典 ：1121543 300
if __name__ == '__main__':

    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt' #239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin' #2s

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    #trans_bin(sql_path,sql_path_bin)
    #trans_bin(ps_path, ps_path_bin)
    #113440 27970(2) 49409(12),50226(30),55993(98)

    #==========================  ==========最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'



    #txt存储数组向量，读取时间：30s,以pickle文件存储0.23s,所以最后采用pkl文件

    #get_new_dict(ps_path_bin,python_word_path,python_word_vec_path,python_word_dict_path)
    #get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================
    #sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'
    #get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    #Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)




    #python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    #python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'
    #get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)

    #处理成打标签的形式
    staqc_python_f ='../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)

    print('序列化完毕')
    #test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)

'''
#修改：删除large相关代码
'''





