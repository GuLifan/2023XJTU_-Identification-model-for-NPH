import cv2
import numpy as np
import torch

# part1: 测量evans指数
def tell_evans(ventricle_img, skull_img) -> float:
    """根据脑室识别结果和颅骨识别结果计算出evans指数

    Args:
        ventricle_img (np.array): 脑室识别结果tensor
        skull_img (np.array): 颅骨识别结果tensor
    
    Returns:
        float: evans指数
    """  
    # 先考察abs(Xmax):
    record_array_1 = []  # 记录每一行“黑色区域”在x方向上的距离
    for i in range(skull_img.shape[0]):
        rec_array_1 = []
        sig = 0 # 标记：看是否能扫到黑色点，如扫不到，[-1]就会出问题
        for j in range(skull_img.shape[1]):
             judge = skull_img[i, j] # (i,j)点的像素值
             if(judge==0): # 说明扫描到了左端的黑点
                rec_array_1.append(j)
                sig = 1
        
        distance_i = 0
        if(sig==1):
            rec_array_1 = np.array(rec_array_1) # 将list转换成ndarray
            rec_array_1.sort()
            distance_i = rec_array_1[-1] - rec_array_1[0] # 对应于rank(i)的“黑色区域在x向的距离”
            record_array_1.append(distance_i)             # 将这个距离加入列表

    record_array_1 = np.array(record_array_1) # 将list转换成ndarray
    record_array_1.sort() # 对所有rank(i)的距离进行排序
    axis_max_1 = record_array_1[-1] # 取的是所有“黑色区域”在x方向上的距离的max（即：x方向上的区域主轴）

    # 再考察abs(Ymax):
    record_array_2 = [] # 记录每一行“黑色区域”在y方向上的距离
    for i in range(skull_img.shape[1]):
        rec_array_2 = []
        sig_1 = 0
        for j in range(skull_img.shape[0]):
            judge = skull_img[j,i] # (i,j)点的像素值
            if(judge==0): # 说明扫描到了上端的黑点
                rec_array_2.append(j)
                sig_1 = 1
        distance_i = 0
        if(sig_1==1):
            rec_array_2 = np.array(rec_array_2)
            rec_array_2.sort()
            distance_i = rec_array_2[-1] - rec_array_2[0] # 对应于rank(i)的“黑色区域在y向的距离”
            record_array_2.append(distance_i)
    
    record_array_2 = np.array(record_array_2)
    record_array_2.sort()
    axis_max_2 = record_array_2[-1]  # 取的是所有“黑色区域”在y向上的距离的max（即：y方向上的区域主轴）

    if(axis_max_1 > axis_max_2):
        # 此时图片横向长度>竖向长度：需要顺时针旋转90度
        skull_img = np.transpose(np.flipud(skull_img))
    else:
        # 此时图片横向长度<竖向长度：不用动
        skull_img = skull_img

#==========================================================================================================

    # (1)ventricle_img 是一个 256x256 的 ndarray
    ventricle_img_array = []                # 记录每一行“黑色脑室”在x方向上的距离
    for i in range(ventricle_img.shape[0]):
        rec_array = []
        flag = 0
        for j in range(ventricle_img.shape[1]):
            judge = ventricle_img[i, j] # (i,j)点的像素值
            if(judge==0): # 说明扫描到了左端的黑点
                rec_array.append(j)
                flag = 1
        distance_i = 0
        if(flag==1):
            rec_array = np.array(rec_array) # 将list转换成ndarray
            rec_array.sort()
            distance_i = rec_array[-1] - rec_array[0]
        ventricle_img_array.append(distance_i)

    ventricle_img_array = np.array(ventricle_img_array) # 将list转换成ndarray
    ventricle_img_array.sort()
    distance_max = ventricle_img_array[-1] # 取的是所有“黑色脑室”在x向上的距离的max



    distance_max_1 = min(axis_max_1,axis_max_2) # 取的是所有“黑色颅骨”在x向上的距离的max

    return distance_max/distance_max_1

# part2: 自动给出诊断意见
def doctor(evans_index: float, age: int, BMI: float, gender: str) -> (bool, str):
    """根据evans指数，年龄，BMI，性别，判断脑子是否有病

    Args:
        evans_index (float): evans指数
        age (int): 年龄
        BMI (float): BMI
        gender (str): 性别P

    Returns:
        (bool, str): 判断结果, 诊断意见
    """    
    # todos
    if evans_index > 0.33:
        return (True,f"寄辣，你可能有脑积水！你的evans指数是{evans_index}")
    else:
        return (False,f"你很安全！你的evans指数是{evans_index}")