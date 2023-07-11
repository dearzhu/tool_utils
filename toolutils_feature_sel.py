'''
@Project ：tool_utils 
@File    ：toolutils_feature_sel.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：11/07/2023 22:28 
'''


def fun_del_loss(data):
    """
    根据缺失值过滤特征
    :param data: 数据源
    :return: 返回缺失率小于90%的字段
    """
    loss_value = data.isna().sum() / data.shape[0]
    del_col = list(loss_value[loss_value >= 0.9].index)
    other_col = list(set(data.columns) - set(del_col))
    print('总共字段个数：{} 个，其中，缺失率》=0.9的个数：{}个，删除：{}个，剩余：{}个'.format(len(loss_value), len(del_col), len(del_col),
                                                                len(other_col)))
    return data[other_col]


def fun_del_unique(data):
    """
    根据唯一值过滤特征
    :param data: 数据源
    :return: 返回唯一值小于90%的数据源
    """
    del_col = []
    for col in data.columns:
        unique_value = data[col].value_counts(normalize=True)
        if len(unique_value[unique_value >= 0.9].index) != 0:
            del_col.append(col)
    other_col = list(set(data.columns) - set(del_col))
    print('总共字段个数：{} 个，其中，唯一值>=0.9的个数：{}个，删除：{}个，剩余：{}个'.format(len(data.columns), len(del_col), len(del_col),
                                                                len(other_col)))
    return data[other_col]
