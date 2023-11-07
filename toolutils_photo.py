'''
@Project ：tool_utils 
@File    ：toolutils_photo.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：02/07/2023 07:40 
'''
import os.path

from PIL import Image


def photo_size_press(path, num1, num2):
    """
    按照照片尺寸压缩图片
    :param path: 文件路径
    :param num1: 压缩宽度比，用来确定压缩后的图片尺寸
    :param num2: 压缩高度比，用来确定压缩后的图片尺寸
    :return:
    """
    # 读入图片
    image = Image.open(path)
    # 缩小图片尺寸
    width, height = image.size
    image = image.resize((int(width / num1), int(height / num2)), Image.ANTIALIAS)  # Image.ANTIALIAS 用来确认图片是否进行锐化
    # 调整图片质量并保存
    dir_, file_name = os.path.split(path)
    # 图片保存
    image.save(os.path.join(dir_, file_name.split('.')[0] + '_ys.' + file_name.split('.')[1]), quality=70)
