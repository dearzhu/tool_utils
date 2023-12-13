'''
@Project ：tool_utils 
@File    ：usually_utils.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：07/11/2023 20:59 
'''

import pandas as pd
import numpy as np


def fun_reduce_mem_usage(df):
    """
    数据压缩
    :param df:
    :return:
    """
    print('*' * 40, '数据压缩')

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dateframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of after optimization is {:.2f} MB'.format(end_mem))
    print('Decreased by  {:.1f} %'.format(100 * (start_mem - end_mem) / start_mem))
    return df


##离散变量与连续变量区分
def fun_category_continue_separation(df, label):
    print('*' * 40, '变量类型选择')
    feature_names = list(df.columns)
    if label in feature_names:
        feature_names.remove(label)
    ##先判断类型，如果是int或float就直接作为连续变量
    numerical_var = list(
        df[feature_names].select_dtypes(
            include=['int', 'int8','int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']).columns)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    print('numerical_var', numerical_var)
    print('categorical_var: {} 个，numerical_var：{} 个'.format(len(categorical_var), len(numerical_var)))
    return categorical_var, numerical_var
