#获取需要训练的变量
import tensorflow as tf

def get_var_list(target_list,var_list):
    for var in var_list:
        for layer in range(len(target_list)):
            kw=target_list[layer]
            if kw in var.name:
                yield var
                break
            else:
                continue