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
def fun_category_continue_separation(df, feature_names_list, label):
    """
    筛选离散连续变量
    :param df:数据源
    :param feature_names_list: 特征字段列表
    :param label:标签值
    :return:离散变量、连续变量
    """
    print('*' * 40, '变量类型选择')
    if label in feature_names_list:
        feature_names_list.remove(label)
    ##先判断类型，如果是int或float就直接作为连续变量
    numerical_var = list(
        df[feature_names_list].select_dtypes(
            include=['int', 'int8', 'int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']).columns)
    categorical_var = [x for x in feature_names_list if x not in numerical_var]
    print('categorical_var: {} 个，numerical_var：{} 个'.format(len(categorical_var), len(numerical_var)))
    return categorical_var, numerical_var


def fun_get_woe_IV(data):
    """
    计算WOE、IV值
    :param data:
    :return:
    """
    good_num = data['good'].sum()
    bad_num = data['bad'].sum()
    data['bad_woe'] = (data['bad'] + 1) / bad_num
    data['good_woe'] = (data['good'] + 1) / good_num
    data['woe_value'] = np.log(data['bad_woe'] / data['good_woe'])
    data['iv_value'] = (data['bad_woe'] - data['good_woe']) * data['woe_value']
    return data, data['iv_value'].sum()


def fun_Cramers_V(data, col, label, cnt_col):
    """
    计算克莱姆法则相关系数，无序的离散变量
    :param data:数据源
    :param col:离散变量
    :param label:因变量
    :param cnt_col:计算数量的字段
    :return:克莱姆相关系数
    """
    df_tmp = pd.pivot_table(data=data, index=label, columns=col, values=cnt_col, aggfunc=len)
    df_tmp_E = pd.DataFrame()
    df_tmp_r = df_tmp.sum(axis=1)
    df_tmp_c = df_tmp.sum(axis=0)
    for index_r in df_tmp_r.index:
        for index_c in df_tmp_c.index:
            df_tmp_E.loc[index_r, index_c] = df_tmp_c[index_c] * df_tmp_r[index_r] / df_tmp_r.sum()
    df_tmp_tj = pd.DataFrame()
    for index_c in df_tmp.index:
        for index_r in df_tmp.columns:
            df_tmp_tj.loc[index_c, index_r] = (df_tmp.loc[index_c, index_r] - df_tmp_E.loc[index_c, index_r]) ** 2 / \
                                              df_tmp_E.loc[index_c, index_r]
    value_ = np.sqrt(df_tmp_tj.sum().sum() / df_tmp.sum().sum() * (min(df_tmp.shape) - 1))
    return value_


value_ = fun_Cramers_V(data=data, label='sex', cnt_col='cnt', col='label')
