#获取一个批次的数据
import random

def next_batch(train_data,train_target,batch_size):
    #内置采样函数，保证采取到不同的行索引
    index=random.sample(range(0,train_target.shape[0]),batch_size)
    batch_data=train_data.iloc[index]
    batch_target=train_target.iloc[index]
    return batch_data,batch_target