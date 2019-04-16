#加载测试数据
import pandas as pd
def load_test_data(data_size):
    i=0
    train_temp = pd.DataFrame()
    train_temp_y = pd.Series()
    for size in data_size:
        # train_temp.append(pd.DataFrame(data=pd.read_csv('fashion_mnist/test/leibie_' + str(i))))
        train_temp = pd.concat([train_temp, pd.DataFrame(data=pd.read_csv('fashion_mnist/test/leibie_' + str(i)))],ignore_index=True)
        train_temp_y = pd.concat([train_temp_y, pd.Series(data=[i for z in range(1000)])], ignore_index=True)
        i += 1
    train_y = pd.get_dummies(train_temp_y)
    train_temp/=256
    return train_temp, train_y