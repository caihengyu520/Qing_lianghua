#以该类传递的数据作为更新集中的数据
from _next_batch import next_batch

class Data:
    def __init__(self,train_x,train_y,batch_size):
        self.train_x=train_x
        self.train_y=train_y
        self.batch_size=batch_size
        self.train_data=None
        self.train_label=None
        self.len=train_y.shape[0]
    def set_data(self):
        self.train_data,self.train_label=next_batch(self.train_x,self.train_y,self.batch_size)
        self.len=self.batch_size
    def get_train_data(self):
        return self.train_data
    def get_train_label(self):
        return self.train_label
    def get_train_len(self):
        return self.len
    def get_data(self,train_data,train_label):
        self.train_data=train_data
        self.train_label=train_label
    def set_len(self):
        self.len=self.train_y.shape[0]