#加载训练数据
import random
import numpy as np
import pandas as pd

def load_test(data_size):
    i=0
    train_temp = pd.DataFrame()
    train_temp_y = pd.Series()
    for size in data_size:
        train_temp = pd.concat([train_temp, pd.DataFrame(data=pd.read_csv('fashion_mnist/test/leibie_' + str(i)))],ignore_index=True)
        train_temp_y = pd.concat([train_temp_y, pd.Series(data=[i for z in range(1000)])], ignore_index=True)
        i += 1
    index=[]
    k=0
    for size in data_size:
        index_=random.sample(range(k*1000,(k+1)*1000),int(size*1000))
        for x in index_:
            index.append(x)
        k+=1
    print(len(index))
    index_=np.reshape(index,len(index))
    # print(collections.Counter(index_))
    data=train_temp.iloc[index_]
    target=train_temp_y.iloc[index_]
    train_data_y = pd.get_dummies(target)
    data/=256
    return data,train_data_y