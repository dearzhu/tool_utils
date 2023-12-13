'''
@Project ：tool_utils 
@File    ：autoChooseVar.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：13/12/2023 22:00 
'''

from usually_utils import fun_reduce_mem_usage
from usually_utils import fun_category_continue_separation


class AutoChooseVar:
    def __init__(self, data, tag, except_cols):
        self.data = data
        self.tag = tag
        self.except_cols = except_cols

    def fun_sel_var(self):
        data = fun_reduce_mem_usage(self.data)
        print(data.info())
        categorical_var, numerical_var = fun_category_continue_separation(df=data,
                                                                          label=self.tag)

