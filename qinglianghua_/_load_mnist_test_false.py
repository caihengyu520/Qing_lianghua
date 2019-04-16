#加载训练数据
import random
import numpy as np
import pandas as pd

def load_mnist_test_false(data_size):
    i=0
    lenth=[]
    train_temp = pd.DataFrame()
    train_temp_y = pd.Series()
    for size in data_size:
        data=pd.read_csv('mnist_false_dataset/shiyan5/leibie_' + str(i)+"/"+str(i)+".csv")
        lent=data.shape[0]
        lenth.append(lent)
        train_temp = pd.concat([train_temp, pd.DataFrame(data=data)],ignore_index=True)
        train_temp_y = pd.concat([train_temp_y, pd.Series(data=[i for z in range(lent)])], ignore_index=True)
        i += 1
    index=[]
    k=0
    len_start=0
    len_end=0
    for size in data_size:
        len_end+=lenth[k]
        index_=random.sample(range(len_start,len_end),int(size*lenth[k]))
        for x in index_:
            index.append(x)
        len_start=len_end
        k+=1
    print(len(index))
    index_=np.reshape(index,len(index))
    # print(collections.Counter(index_))
    data=train_temp.iloc[index_]
    target=train_temp_y.iloc[index_]
    train_data_y = pd.get_dummies(target)
    return data,train_data_y