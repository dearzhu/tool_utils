'''
@Project ：tool_utils 
@File    ：toolutils_file.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：01/07/2023 22:24 
'''

import os


def file_zip(path):
    """
    遍历压缩该目录下所有文件，包括子文件夹。所有的压缩文件放在该目录下
    :param path: 需要压缩的文件路径
    :return:
    """
    for i, j, k in os.walk(path):
        if 'ipynb_checkpoints' not in i:
            if os.path.isdir(i) == True:
                if i == path:
                    print(i)
                    os.system('zip -l {}.zip  ./*'.format(os.path.split(i)[1]))
                    # print('zip -l {}.zip  ./*'.format(os.path.split(i)[1]))
                else:
                    # print('zip -l {}.zip  {}/*'.format(i.replace(path+'/','').replace('/','_'),i.replace(path+'/','')))
                    os.system('zip -l {}.zip  {}/*'.format(i.replace(path + '/', '').replace('/', '_'),
                                                           i.replace(path + '/', '')))
