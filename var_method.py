'''
@Project ：tool_utils 
@File    ：var_method.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：13/07/2023 21:04 
'''
import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor


def fun_get_cols_vif(x_train):
    """

    :param x_train:
    vif_value:🧍各个变量的vif 值
    vars_vif：vif 小于10的变量
    :return:
    """
    vif = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[0])]
    vif_value = pd.DataFrame(zip(x_train.columns.tolist(), vif), columns=['var_name', 'vif']).sort_values(by='vif',
                                                                                                          ascending=False).reset_index(
        drop=True)
    vars_vif = vif_value.loc[vif_value['vif'] < 10, 'var_name'].tolist()
    return vif_value, vars_vif


def fun_proceed_vif_choose_vars(df, num_cols, excepts_cols, silent=True):
    """

    :param df:数据源
    :param num_cols：连续型变量列表，用来检验vif的字段
    :param excepts_cols: 不用进行vif 检验的字段
    :param silent: 是否打印每个字段的vif
    :return:
    """
    tmp_df = df.copy()
    vif_cols = list((set(num_cols) - set(excepts_cols)) & set(tmp_df.columns))
    cols_vif_value, choosed_cols = fun_get_cols_vif(tmp_df[vif_cols])
    if not silent:
        print(cols_vif_value)
    return choosed_cols
