import numpy as np
import random

def next_batch_(train_data,train_target,batch_size,start,index_list):
    if start+batch_size>len(index_list):
        start=0
        random.shuffle(index_list)
        # print('random.shuffle is start: ',index_list)
    index_=[index_list[i] for i in range(start,start+batch_size)]
    batch_data=train_data.iloc[index_]
    batch_target=train_target.iloc[index_]
    start+=batch_size
    return start,index_list,batch_data,batch_target