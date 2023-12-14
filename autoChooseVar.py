'''
@Project ：tool_utils 
@File    ：autoChooseVar.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：13/12/2023 22:00 
'''
from usually_utils import fun_get_woe_IV
from usually_utils import fun_reduce_mem_usage
from usually_utils import fun_category_continue_separation
from variable_bin_methods import cal_bin_value

from variable_bin_methods import cont_var_bin, cont_var_bin_map
import pandas as pd


class AutoChooseVar:
    def __init__(self, data, tag, except_cols, IV_flag, method):
        self.data = data
        self.tag = tag
        self.except_cols = except_cols
        self.IV_flag = IV_flag
        self.method = method

    def fun_sel_var(self):
        print('总共变量：{} 个，需要剔出的变量：{} 个，参与计算的变量：{} 个'.format(len(self.data.columns), len(self.except_cols),
                                                           len(self.data.columns) - len(self.except_cols)))
        data = fun_reduce_mem_usage(self.data)
        categorical_var, numerical_var = fun_category_continue_separation(df=data,
                                                                          label=self.tag)
        if self.IV_flag == True:

            print("*" * 40, 'IV值计算开始')
            WOE_dict = {}
            IV_dict = {}
            print("*" * 20, '离散变量IV值计算')

            for col in set(categorical_var) - set(self.except_cols):  # 提出掉不需要考虑的变量
                df_temp = cal_bin_value(x=data[col], y=data[self.tag], bin_min_num_0=10)
                woe, iv = fun_get_woe_IV(df_temp)
                WOE_dict[col] = woe
                IV_dict[col] = iv
            print("*" * 20, '连续变量IV值计算')
            dict_cont_bin = {}
            print("*" * 15, '连续变量分箱')
            for i in numerical_var:
                dict_cont_bin[i], gain_value_save, gain_rate_save = cont_var_bin(data[i],
                                                                                 data[self.tag],
                                                                                 method=self.method,
                                                                                 mmin=4,
                                                                                 mmax=10,
                                                                                 bin_rate=0.01,
                                                                                 stop_limit=0.1,
                                                                                 bin_min_num=20)
            df_cont_bin_train = pd.DataFrame()
            print("*" * 15, '连续变量分箱映射')
            for i in dict_cont_bin.keys():
                df_cont_bin_train = pd.concat([df_cont_bin_train, cont_var_bin_map(data[i], dict_cont_bin[i])],
                                              axis=1)
            df_cont_bin_train[self.tag] = data[self.tag]
            for col in dict_cont_bin.keys():
                df_temp = df_cont_bin_train.groupby(by=[col + '_BIN', self.tag])[
                    self.tag].count().unstack().reset_index().fillna(0)
                df_temp.rename(columns={0: 'good', 1: 'bad'}, inplace=True)
                woe, iv = fun_get_woe_IV(df_temp)
                WOE_dict[col] = woe
                IV_dict[col] = iv
            print(IV_dict)
            IV_save_var = [key for key in IV_dict.keys() if IV_dict[key] > 0.005]
            print('累计参与变量：{} 个，IV值筛选剔除变量：{} 个，保留变量：{} 个'.format(
                len(categorical_var) + len(numerical_var) - len(self.except_cols),
                len(categorical_var) + len(numerical_var) - len(self.except_cols) - len(IV_save_var), len(IV_save_var)))

        # 方差选择
        # VIF选择
        # 特征重要性选择
