'''
@Project ：tool_utils 
@File    ：toolutils_feature_sel.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：11/07/2023 22:28 
'''
import numpy as np


def fun_del_loss(data):
    """
    根据缺失值过滤特征
    :param data: 数据源
    :return: 返回缺失率小于90%的字段
    """
    loss_value = data.isna().sum() / data.shape[0]
    del_col = list(loss_value[loss_value >= 0.9].index)
    other_col = list(set(data.columns) - set(del_col))
    print('总共字段个数：{} 个，其中，缺失率>=0.9的个数：{}个，删除：{}个，剩余：{}个'.format(len(loss_value), len(del_col), len(del_col),
                                                                len(other_col)))
    return other_col


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
    return other_col


from sklearn.feature_selection import VarianceThreshold


def fun_var_feature_sel(data):
    """
    方差选择特征
    :param data: 连续性变量
    :return: 剩余字段
    """
    var_model = VarianceThreshold(threshold=0.01)
    var_model.fit(data)
    save_index = var_model.get_support(True)
    other_col = [data.columns[i] for i in save_index]
    return other_col


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import networkx as nx
import matplotlib.pyplot as plt


def fun_solve_IV(data, col_list1, label):
    woe_dict = {}
    iv_dict = {}
    if label in col_list1:
        col_list1.remove(label)
    good_cnt = data[data['label'] == 0].shape[0]
    bad_cnt = data[data['label'] == 1].shape[0]
    for col in col_list1:
        value = data.groupby(by=[col, label])[label].count().unstack().reset_index().fillna(0)
        value.rename(columns={0: 'good', 1: 'bad'}, inplace=True)
        value['bad_woe'] = (value['bad'] + 1) / bad_cnt
        value['good_woe'] = (value['good'] + 1) / good_cnt
        value['woe_value'] = np.log(value['bad_woe'] / value['good_woe'])
        value['iv_value'] = (value['bad_woe'] - value['good_woe']) * value['woe_value']
        woe_dict[col] = value
        iv_dict[col] = value['iv_value'].sum()
    return woe_dict, iv_dict


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


def fun_chose_corr_graph(df, to_num_cols, corr_threshold=0.6, metric_dict={}, only_choose_one=False, slient=False):
    to_num_cols = list(set(to_num_cols) & set(df.columns))
    to_num_cols = list(set(to_num_cols) & set(metric_dict.keys()))
    df_origin = df[to_num_cols].copy()
    cm_df = df_origin.copy()
    # 生成网络图时的相关性阈值
    cm_df = cm_df.reset_index()
    corr_df = {'c1': [], 'c2': [], 'corr': []}
    for c1 in list(cm_df['index']):
        for c2 in list(cm_df.columns):
            if c2 != 'index' and c1 != c2 and not (c2 in corr_df['c1'] and c1 in corr_df['c2']):
                corr_df['c1'].append(c1)
                corr_df['c2'].append(c2)
                corr_df['corr'].append(cm_df.loc[cm_df['index'] == c1, c2].values[0])
    corr_df = pd.DataFrame(corr_df)
    corr_df = corr_df[corr_df['corr'] > corr_threshold].copy().reset_index()
    del corr_df['index']
    data = corr_df.copy()
    del corr_df
    node_list = data['c1'].append(data['c2']).unique()
    G = nx.Graph()
    G.add_nodes_from(node_list)
    for i in range(data.shape[0]):
        if i % 1000 == 0:
            if not slient:
                print('---------------------编号：', i, '--------------------')
        G.add_edge(data['c1'][i], data['c2'][i], weight=data['corr'][i])
    pos = nx.spring_layout(G)
    # 输出各个强关联变量是图
    if not slient:
        print('------------------生成联通子图------------------------')
    subgraph_list = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    subgraph_len = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # 图中点的信息字典,说明图中点属于哪个图
    g_node_info = dict()
    graphs_nodes = []
    choosed_nodes = []
    if not slient:
        print('子图数量：', len(subgraph_list))
        print('---------------遍历每个子图---------------------')
    for i in range(len(subgraph_list)):
        sub_graph = G.subgraph(subgraph_list[i])
        sub_graph_len = subgraph_len[i]
        sub_graph_df = data[(data['c1'].isin([sub_id for sub_id in subgraph_list[i]]))].copy()
        if not slient:
            print('子图规模：', sub_graph_len)
        sg = list(sub_graph.nodes)
        graphs_nodes = graphs_nodes + sg
        graph_cols = {}
        for c in sg:
            if str(c) in metric_dict.keys():
                graph_cols[c] = metric_dict[c]
        if only_choose_one:
            graph_cols = [sorted(zip(graph_cols.values(), graph_cols.keys()), reverse=True)[0]]
        else:
            if len(sg) >= 10:
                print(graph_cols)
                graph_cols = sorted(zip(graph_cols.values(), graph_cols.keys()), reverse=True)[0:4]
            elif len(sg) >= 5:
                graph_cols = sorted(zip(graph_cols.values(), graph_cols.keys()), reverse=True)[0:3]
            elif len(sg) >= 3:
                graph_cols = sorted(zip(graph_cols.values(), graph_cols.keys()), reverse=True)[0:2]
            else:
                graph_cols = [sorted(zip(graph_cols.values(), graph_cols.keys()), reverse=True)[0]]
        graph_col_names = []
        for v in graph_cols:
            v0, v1 = v
            graph_col_names.append(v1)
        choosed_nodes = choosed_nodes + graph_col_names
        corr_matrix = pd.pivot_table(data[(data['c1'].isin(sg)) & (data['c2'].isin(sg))], index='c1', columns='c2',
                                     values='corr', aggfunc='mean')
        # 如果是不输出信息的模式
        if not slient:
            plt.figure(figsize=[10, 6])
            nx.draw(sub_graph, nx.spring_layout(sub_graph), node_color='red', with_labels=True, edge_color='black',
                    node_size=50, font_size=16)
            plt.show()
    print('全部变量数量：', len(metric_dict.keys()))
    print('按照强相关性阈值构造强相关性关系网中的变量数量：', len(graphs_nodes))
    print('选出强相关性关系网中的变量数量：', len(list(choosed_nodes)))
    print('最终选出的变量数量：', len(list(set(metric_dict.keys()) - set(graphs_nodes)) + list(choosed_nodes)))
    return list(set(metric_dict.keys()) - set(graphs_nodes)) + list(choosed_nodes)
