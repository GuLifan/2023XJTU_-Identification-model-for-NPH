# @file final.py
# Copyright (c) 谷立帆 胡博瑄 刘承臻 2024
# @brief 这是用来上交软著的代码, 要把我们目前所有的代码都复制到这个里面去import glob
import numpy as np
import torch
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from unet.unet_model import UNet
from utils.args import parser
from config.train_params import CATEGORY
import torch.nn as nn
from utils.dataset import Img_Loader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pprint import pprint
from tqdm import tqdm
from config.train_params import CATEGORY
import gradio as gr
import io
from PIL import Image
from utils.doctor import tell_evans, doctor, nii_file_slicer
import torch.nn.functional as F
from train import train_net
from predict import predict, eval
import glob
import json
import random
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image, ImageOps
import argparse
CATEGORY = "ventricle"
MODEL_TYPE = "unet"
EPOCHS = 100
BATCH_SIZE = 20
PICTURE_SIZE = 256
IMG_SIZE_1 = 20
IMG_SIZE_2 = 30
IMG_SIZE_3 = 40
IMG_SIZE_4 = 50
IMG_SIZE_5 = 60
TUNE_SIZE_1 = 20
TUNE_SIZE_2 = 30
TUNE_SIZE_3 = 40
TUNE_SIZE_4 = 50
TUNE_SIZE_5 = 60
AS = 3
LR = 0.1
SCALE = 5
DATA_PATH = f"./data/{CATEGORY}/train"
OUTPUT_PATH = "./output"
THRESHOLD = 0.5
DOCTOR = 0
WEB_SIZE = 256
NEW = 0
ACCURACY_1 = 0.95
ACCURACY_2 = 0.90
ACCURACY_3 = 0.85
ACCURACY_4 = 0.80
WINDOW_1 = 0
WINDOW_2 = 0
WINDOW_3 = 0
WINDOW_4 = 0
# 定义Unet模型本体
class UNet(nn.Module):
    # 函数初始化
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
    # 前向传播函数
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    # 检查点函数
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)   
# 定义双卷积层
# innovation for Unet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # 如果没有就更新
        if not mid_channels:
            mid_channels = out_channels
        # 正常操作
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 这里采用的激活函数是 ReLU
    # 当前处理完毕，进入下一层
    def forward(self, x):
        return self.double_conv(x)
from config.train_params import (
    CATEGORY,
    MODEL_TYPE,
    EPOCHS,
    BATCH_SIZE,
    LR,
    SCALE,
    DATA_PATH,
    OUTPUT_PATH,
    THRESHOLD,
)
# 模型参数引入
parser = argparse.ArgumentParser(description="训练模型相关参数")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="训练时batch_size")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练时epochs")
parser.add_argument("--lr", type=float, default=LR, help="学习率")
parser.add_argument("--data-path", type=str, default=DATA_PATH, help="数据集路径")
parser.add_argument("--output-path", type=str, default=OUTPUT_PATH, help="模型输出路径")
parser.add_argument("--scale", type=int, default=SCALE, help="图片像素值缩放大小")
parser.add_argument("--threshold", type=int, default=THRESHOLD, help="模型判别阈值")
parser.add_argument("--model-type", type=str, default=MODEL_TYPE, help="模型类型,可选unet、resnet18等")
parser.add_argument("--category", type=str, default=CATEGORY, help="模型类别,可选ventricle、skull等")
# 设计引入图片的函数
class Img_Loader(Dataset):
    # 函数初始化以实现复用
    # 1. 图像路径
    # 2. 图像缩放因子
    # 3. 图像格式
    def __init__(self, root_dir: str, scale: int = 5):
        self.root_dir = root_dir
        self.imgs_path = glob.glob(os.path.join(self.root_dir, "image/*.jpg"))
        self.scale = scale
    # 图像增强
    def augment(self, image, option: int):
        """使用cv2.flip()进行图像增强
        Args:
            image : 待增强的图片
            option (int): 增强选项, -1=中心旋转180, 0=上下翻转, 1=左右翻转
        Returns:
            image: 增强后的图片
        """
        fliped = cv2.flip(image, option)
        return fliped
    # 返回函数
    def __getitem__(self, index):
        """返回索引对应的图片和标签
        Args:
            index (int): 寻找的索引
        Returns:
            image, label: 索引对应的图片和标签
        """
        image_path = self.imgs_path[index]
        label_path = image_path.replace("image", "label")
        label_path = label_path.replace(".jpg", "_label.png")
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将图片转为单通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 反转标签颜色, 需要加深特征量, 使其更明显
        label = 255 - label
        # 脑室和一部分非脑室的区域有相似的色深, 为了使得标签给出的区域权重加大, 故对其像素值进行缩放
        label = label * self.scale
        # 归一化
        if label.max() > 1:
            label = label / 255.0
        # 随机增强
        option = random.choice([-1, 0, 1, 2])
        if option != 2:
            image = self.augment(image, option)
            label = self.augment(label, option)
        return image, label
    # 汇报函数，反应数据集的大小
    def __len__(self):
        """返回数据集的大小
        Returns:
            int: 数据集的大小
        """
        return len(self.imgs_path)
# 首次激活调用
if __name__ == "__main__":
    # 引入图片库，这里采用相对路径以实现可复用性
    img_lib = Img_Loader("./data/ventricle/train")
    print("Loaded data counts: ", len(img_lib))
    # 模型加载
    traner_loader = torch.utils.data.DataLoader(
        dataset=img_lib,
        batch_size=1,
        shuffle=True,
    )
    # 打印函数
    # 注意要以元组形式
    for img, lable in traner_loader:
        print(img.shape)
        # cv2.imwrite("./data/train/tmp/tmp.png", np.ndarray(lable[0]))
# part1: 测量evans指数
def tell_evans(ventricle_img, skull_img) -> float:
    """根据脑室识别结果和颅骨识别结果计算出evans指数
    Args:
        ventricle_img (np.array): 脑室识别结果array
        skull_img (np.array): 颅骨识别结果array
    Returns:
        float: evans指数
        float: 脑室长度
        float: 颅骨长度
    """
    # 先考察abs(Xmax):
    record_array_1 = []  # 记录每一行“黑色区域”在x方向上的距离
    for i in range(skull_img.shape[0]):
        rec_array_1 = []
        sig = 0  # 标记：看是否能扫到黑色点，如扫不到，[-1]就会出问题
        for j in range(skull_img.shape[1]):
            judge = skull_img[i, j]  # (i,j)点的像素值
            # 说明扫描到了左端的黑点
            if judge == 0:  
                rec_array_1.append(j)
                sig = 1
        distance_i = 0
        if sig == 1:
            rec_array_1 = np.array(rec_array_1)  # 将list转换成ndarray
            rec_array_1.sort()
            # 对应于rank(i)的“黑色区域在x向的距离”
            distance_i = (
                rec_array_1[-1] - rec_array_1[0]
            )  
            # 将这个距离加入列表
            record_array_1.append(
                distance_i
            )  
    # 将list转换成ndarray
    record_array_1 = np.array(
        record_array_1
    )  
    # 对所有rank(i)的距离进行排序
    record_array_1.sort()  
    # 取的是所有“黑色区域”在x方向上的距离的max（即：x方向上的区域主轴）
    axis_max_1 = record_array_1[
        -1
    ]  
    # 再考察abs(Ymax):
    record_array_2 = []  
    # 记录每一行“黑色区域”在y方向上的距离:
    for i in range(skull_img.shape[1]):
        rec_array_2 = []
        sig_1 = 0
        for j in range(skull_img.shape[0]):
            # (i,j)点的像素值
            judge = skull_img[j, i]  
            if judge == 0:  
                # 说明扫描到了上端的黑点
                rec_array_2.append(j)
                sig_1 = 1
        distance_i = 0
        if sig_1 == 1:
            rec_array_2 = np.array(rec_array_2)
            rec_array_2.sort()
            # 对应于rank(i)的“黑色区域在y向的距离”
            distance_i = (
                rec_array_2[-1] - rec_array_2[0]
            )  
            record_array_2.append(
                distance_i
            )
    record_array_2 = np.array(
        record_array_2
    )
    record_array_2.sort()
    # 取的是所有“黑色区域”在y向上的距离的max
    # (即：y方向上的区域主轴)
    axis_max_2 = record_array_2[
        -1
    ]  
    if axis_max_1 > axis_max_2:
        # 此时图片横向长度>竖向长度：需要顺时针旋转90度
        skull_img = np.transpose(np.flipud(skull_img))
    else:
        # 此时图片横向长度<竖向长度：不用动
        skull_img = skull_img
    # (1) ventricle_img 是一个 256x256 的 ndarray
    # 记录每一行“黑色脑室”在x方向上的距离
    ventricle_img_array = []  
    for i in range(ventricle_img.shape[0]):
        rec_array = []
        flag = 0
        for j in range(ventricle_img.shape[1]):
            # (i,j)点的像素值
            judge = ventricle_img[i, j]  
            if judge == 0:  
                # 说明扫描到了左端的黑点
                rec_array.append(j)
                flag = 1
        distance_i = 0
        if flag == 1:
            # 将list转换成ndarray
            rec_array = np.array(rec_array)  
            rec_array.sort()
            distance_i = rec_array[-1] - rec_array[0]
        ventricle_img_array.append(
            distance_i
        )
    ventricle_img_array = np.array(
        ventricle_img_array
    )  
    # 将list转换成ndarray
    ventricle_img_array.sort()
    # 取的是所有“黑色脑室”在x向上的距离的max
    distance_max = ventricle_img_array[-1]  
    # 取的是所有“黑色颅骨”在x向上的距离的max
    distance_max_1 = min(
        axis_max_1, axis_max_2
    )  
    return distance_max / distance_max_1, distance_max, distance_max_1
# part2: 自动给出诊断意见
def doctor(
    evans_index: float, age: int, stature: float, weight: float, gender: str
) -> (bool, str):
    """
    根据evans指数,年龄,BMI,性别,判断是否有病
    Args:
        evans_index (float): evans指数
        age (int): 年龄
        stature (float): 身高
        weight (float): 体重
        gender (str): 性别
    Returns:
        (bool, str): 判断结果, 诊断意见
    PS:
    医师意见: Utils/doctor.py
    """
    # todos
    if evans_index > 0.33:
        return (True, f"你可能有脑积水! 你的evans指数是{evans_index}")
    else:
        return (False, f"你很安全! 你的evans指数是{evans_index}")
def nii_file_spliter(file_path: str) -> str:
    try:
        # 读取nifti文件头信息
        nifti_img = nib.load(file_path)
        header = nifti_img.header
        # 获取维度顺序
        dim_order = header.get_data_shape()
        if len(dim_order) == 3:
            # 三维数据, 直接返回
            return file_path
        elif len(dim_order) == 4:
            # 四维数据, 读取第一层
            data = nifti_img.get_fdata()
            num_timepoints = data.shape[3]
            new_data = data[:, :, :, 0]
            new_img = nib.Nifti1Image(new_data, nifti_img.affine, nifti_img.header)
            # 就地修改文件
            nib.save(new_img, file_path)
            return file_path
    except Exception as e:
        print(e)
def nii_file_slicer(file_path: str):
    file_path = nii_file_spliter(
        file_path
    )
    nii_image = nib.load(
        file_path
    )
    data = nii_image.get_fdata()
    slice_count = data.shape[1]
    for i in range(slice_count):
        slice_data = data[:, i, :]
        if np.nanmin(slice_data) < np.nanmax(slice_data):
            # 归一化
            slice_data = ((slice_data - np.nanmin(slice_data)) / (np.nanmax(slice_data) - np.nanmin(slice_data)) * 255).astype(np.uint8)
        else:
            slice_data = np.zeros_like(slice_data, dtype=np.uint8)
        img = Image.fromarray(slice_data)
        img = ImageOps.pad(img, (256, 256), color='black')
        # convert PIL Image back to np.ndarray
        yield np.array(img)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    # 前向传播计算
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, 
                mode='bilinear', 
                align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, 
                out_channels, 
                in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, 
                in_channels // 2, 
                kernel_size=2, 
                stride=2
            )
            self.conv = DoubleConv(
                in_channels, 
                out_channels
            )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1
        )
    # 前向传播计算
    def forward(self, x):
        return self.conv(x)
# shared
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for train
# The path of the data: 
data_path = "./data/ventricle/train"
output_path = "./output"
# for predict and eval
sources_path = glob.glob("data/ventricle/train/image/*.jpg")
labels_path = glob.glob("data/ventricle/train/label/*.png")
def fine_tune():
    # perflog
    max_batch = 51
    train_perflog = []
    # initialization
    best_record = {"avg_accuracy": 0}
    for epochs in range(100, 310, 10):
        for scale in range(1, 21):
            for lr in np.arange(0.0001, 0.011, 0.0004):
                for batch_size in range(1, max_batch):
                    net = UNet(n_channels=1, 
                               n_classes=1, 
                               bilinear=False
                               )
                    net.to(device)
                    pprint(
                        f"Train params: epochs={epochs}, batch_size={batch_size}, lr={lr}, scale={scale}"
                    )
                    # 自动错误检验
                    try:
                        loss = train_net(
                            net,
                            device,
                            data_path,
                            output_path,
                            epochs,
                            batch_size,
                            lr,
                            scale,
                        )
                    except:
                        pprint("Train failed")
                        # train失败了大概率是因为爆显存了
                        # 此处调小显存, 直接开始新一轮循环
                        max_batch = batch_size - 1
                        break
                    # 对阈值进行筛选
                    for threshold in np.arange(0.1, 2.1, 0.1):
                        pprint(f"|----Predict param: threshold={threshold}")
                        try:
                            predict(net, device, sources_path, threshold)
                            avg_accuracy = eval(sources_path, labels_path)
                            pprint(
                                f"|    |----Result: loss={loss}, accuracy={avg_accuracy}"
                            )
                        except:
                            pprint("|    |----Predict failed")
                            loss = 1
                            avg_accuracy = 0
                        # 记录函数
                        record = {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "scale": scale,
                            "threshold": threshold,
                            "loss": loss.item(),
                            "avg_accuracy": avg_accuracy,
                        }
                        if record["avg_accuracy"] > best_record["avg_accuracy"]:
                            best_record = record
                        train_perflog.append(record)
                    # del net
                    json.dump(
                        train_perflog, open("train_perflog.json", "w"), indent=4, ensure_ascii=False
                    )
                    json.dump(best_record, open("best_record.json", "w"), indent=4, ensure_ascii=False)
if __name__ == "__main__":
    fine_tune()
# 模型预测函数
def predict(net, device, sources_path, threshold=0.5):
    for test_path in sources_path:
        save_res_path = test_path.replace("train/image", "res")
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为原图大小的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # print(pred)
        # 处理结果
        pred[pred >= threshold] = 255
        pred[pred < threshold] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)
def tell_diff(label_path, pred_path):
    # 检验区别
    label = cv2.imread(label_path)
    pred = cv2.imread(pred_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    # 为了使所有数据在RGB同一范围内
    label = 255 - label
    # 纯色惩罚, 仅当标签纯色时返回1, 仅当结果纯色时返回0
    if label.max() <= 0 or label.min() >= 255:
        return 1
    if pred.max() <= 0 or pred.min() >= 255:
        return 0
    accuracy = ssim(label, pred)
    return accuracy
def eval(sources_path, labels_path):
    sum_acc = 0
    for i in range(len(sources_path)):
        label_path = labels_path[i]
        pred_path = sources_path[i].replace("train/image", "res")
        sum_acc += tell_diff(label_path, pred_path)
    acc = sum_acc / len(sources_path)
    # print(acc)
    return acc
if __name__ == "__main__":
    arg = parser.parse_args()
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(
        torch.load(f"./output/{arg.model_type}-{CATEGORY}.pth", map_location=device)
    )
    # 测试模式
    net.eval()
    # 读取所有图片路径
    sources_path = glob.glob(f"data/{CATEGORY}/train/image/*.jpg")
    labels_path = glob.glob(f"data/{CATEGORY}/train/label/*.png")
    # 根据现有进行预测
    predict(net, device, sources_path)
    print("avg_accuracy", eval(sources_path, labels_path))
# 模型训练函数
def train_net(
    net,
    device,
    data_path,
    output_path,
    epochs=50,
    batch_size=1,
    lr=0.001,
    scale=5,
    model_type="unet",
):
    output_model = os.path.join(output_path, f"{model_type}-{CATEGORY}.pth")
    # 数据集
    img_dataset = Img_Loader(data_path, scale)
    # 加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=img_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    # 进行最优化处理
    optimizer = optim.Adam(net.parameters(), lr=lr)
    critirion = nn.BCEWithLogitsLoss()  
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    best_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch", colour="blue"):
        # 开启训练模式
        net.train()
        for img, label in train_loader:
            optimizer.zero_grad()
            # 全精度训练
            img = img.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            # 选择损失最小的模型保存
            predict = net(img)
            loss = critirion(predict, label)
            # 每次遍历，从而取得整体最小值
            if loss < best_loss:
                best_loss = loss
                torch.save(
                    net.state_dict(), 
                    output_model
                )
            # 更新参数
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
        # print(f"#{epoch+1} loop. 
        # Loss rate: ", loss.item())
    # print(f"\nTraining finished. 
    # Best loss: {best_loss}")
    return best_loss.item()
if __name__ == "__main__":
    arg = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始bilinear需置为False
    # 否则从本地加载模型时会出错
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.to(device)
    print(
        train_net(
            net=net,
            device=device,
            data_path=arg.data_path,
            output_path=arg.output_path,
            epochs=arg.epochs,
            batch_size=arg.batch_size,
            lr=arg.lr,
            scale=arg.scale,
            model_type=arg.model_type
        )
    )
# 检验前错函数
try:
    from config.train_params import THRESHOLD
except:
    THRESHOLD = 0.5
# 使用指南
"""
使用步骤:
1. 上传患者颅内影像图片, 或者是患者颅内影像文件(.nii文件). 注意: **只需要上传二者之一即可**
2. 填写患者信息, 包括姓名, 年龄, 身高, 体重, 性别等
3. 填写医生诊断意见
4. 点击 “开始诊断” 按钮
诊断结果和细节将会出现在按钮下方
"""
# 诊断报告模板
"""
# 诊断报告
## 患者信息
- 患者姓名: {name}
- 患者年龄: {age}
- 患者身高: {stature}
- 患者体重: {weight}
- 患者性别: {gender}
## 诊断结果
- EI指数: {evans_index}
- 脑室宽度: {ventricle_len}
- 颅骨宽度: {skull_len}
## 诊断意见
- AI意见: {ai_diagnosis}
- 医生意见: {doc_opinion}
"""
# 加载本地模型
# 加载方式定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 通道数确立
ventricle_unet = UNet(n_channels=1, n_classes=1)
# 指定加载内容
ventricle_unet.to(device=device)
ventricle_unet.load_state_dict(
    torch.load(f"./output/unet-ventricle.pth", map_location=device)
)
ventricle_unet.eval()
skull_unet = UNet(n_channels=1, n_classes=1)
skull_unet.to(device=device)
skull_unet.load_state_dict(torch.load(f"./output/unet-skull.pth", map_location=device))
skull_unet.eval()
def diagnose(input_img, input_age, input_stature, input_weight, input_gender):
    """根据输入的图像, 年龄, 身高, 体重, 性别等进行诊断
    Args:
        input_img (numpy.array): 输入的灰度图像数组
        input_age (int): 输入的年龄
        input_stature (float): 输入的身高
        input_weight (float): 输入的体重
        input_gender (str): 输入的性别
    Returns:
        Image: 脑室识别图像
        Image: 颅骨识别图像
        str: 诊断意见
        float: Evans指数
        float: 脑室长度
        float: 颅骨长度
    """
    original_image = Image.fromarray(input_img)
    img = input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # 识别脑室
    ventricle_img = ventricle_unet(img_tensor)
    ventricle_img = np.array(ventricle_img.data.cpu()[0])[0]
    ventricle_img[ventricle_img >= THRESHOLD] = 255
    ventricle_img[ventricle_img < THRESHOLD] = 0
    # print("******预测成功******")
    ventricle_image = Image.fromarray(ventricle_img)
    ventricle_image = ventricle_image.convert("RGB")
    # 识别颅骨
    skull_img = skull_unet(img_tensor)
    skull_img = np.array(skull_img.data.cpu()[0])[0]
    skull_img[skull_img >= THRESHOLD] = 255
    skull_img[skull_img < THRESHOLD] = 0
    skull_image = Image.fromarray(skull_img)
    skull_image = skull_image.convert("RGB")
    # 完成诊断函数后取消下面这段的注释
    evans_index, ventricle_len, skull_len = tell_evans(ventricle_img, skull_img)
    ill, diagnosis = doctor(
        evans_index, input_age, input_stature, input_weight, input_gender
    )
    print(f"***evans: {evans_index}")
    indicator = (255, 0, 0) if ill else (0, 255, 0)  # 绿色健康, 红色有病
    # 将输出图像染色
    width, height = ventricle_image.size
    # 遍历图像的每个像素
    # ventricle
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = ventricle_image.getpixel((x, y))
            # 如果像素是黑色，则将其改为指示色
            if r == 0 and g == 0 and b == 0:
                ventricle_image.putpixel((x, y), indicator)
    # skull
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = skull_image.getpixel((x, y))
            # 如果像素是黑色，则将其改为指示色
            if r == 0 and g == 0 and b == 0:
                skull_image.putpixel((x, y), indicator)
    # print("******染色成功******")
    # 叠加
    ventricle_image.paste(original_image, (0, 0), original_image)
    skull_image.paste(original_image, (0, 0), original_image)
    # print("******叠加成功******")
    return (
        ventricle_image,
        skull_image,
        diagnosis,
        evans_index,
        ventricle_len,
        skull_len,
    )
# 数据可视化
# 我们采用Gradio进行网页可视化演示
# 以下是gradio webui组件
guideline = gr.Markdown(value=GUIDELINE, label="使用指南")
# 用户自定义传入图像
input_img = gr.Image(
    image_mode="L",
    sources=["upload"],
    type="numpy",
    label="患者颅内影像",
)
# 用户自定义传入文件
input_file = gr.File(label="患者颅内影像文件")
# 用户信息录入：姓名 / 年龄 / 身高 / 体重 / 性别 / BMI
input_name = gr.Textbox(value="示例", 
                        label="患者姓名", 
                        interactive=True
                )
input_age = gr.Number(minimum=0, 
                      maximum=100, 
                      value=20, 
                      label="患者年龄", 
                      interactive=True
)
input_stature = gr.Number(minimum=0, 
                          maximum=250, 
                          value=170, 
                          label="患者身高(cm)", 
                          interactive=True
)
input_weight = gr.Number(minimum=0, 
                         maximum=200, 
                         value=60, 
                         label="患者体重(kg)", 
                         interactive=True
)
input_gender = gr.Dropdown(
    choices=["男性", "女性"], value="男性", label="患者性别", interactive=True
)
input_doctor = gr.Textbox(label="医生意见", value="Utils/doctor.py", interactive=True)
# 输出文件
output_img_ventricle = gr.Image(
    type="pil",
    image_mode="RGB",
    label="脑室识别结果",
    show_label=True,
    show_share_button=True,
)
output_img_skull = gr.Image(
    type="pil",
    image_mode="RGB",
    label="颅骨识别结果",
    show_label=True,
    show_share_button=True,
)
output_diagnosis = gr.Textbox(value="!!!", label="诊断意见")
report = gr.Markdown(value="诊断报告", label="诊断报告")
error_box = gr.Textbox(label="出错!", visible=False)
diagnose_btn = gr.Button(value="开始诊断", size="lg")
# 插件管理
with gr.Blocks(title="UNet颅内影像识别") as demo:
    guideline.render()
    with gr.Column() as input_panel:
        with gr.Row() as input_sources:
            input_img.render()
            input_file.render()
            with gr.Column() as input_details:
                input_name.render()
                input_age.render()
                input_stature.render()
                input_weight.render()
                input_gender.render()
        input_doctor.render()
        diagnose_btn.render()
    error_box.render()
    with gr.Column(visible=False) as output_panel:
        output_diagnosis.render()
        with gr.Row() as output_alts:
            with gr.Accordion(label="诊断报告") as output_report:
                report.render()
            with gr.Accordion(label="区域识别") as output_images:
                with gr.Row():
                    output_img_ventricle.render()
                    output_img_skull.render()
    def content_wrapper(
        input_name,
        input_img,
        input_file,
        input_age,
        input_stature,
        input_weight,
        input_gender,
        input_doctor,
    ):
        """这是负责处理输入和包装输出的函数
        1. 在调用模型进行推理前, 需要判断输入是否合法, 以及根据输入类型(图片/文件)来选择处理方法; 
        2. 在调用模型推理后, 需要根据模型结果更新WebUI中各个组件的状态和值
        Args:
            input_name (str): 患者姓名
            input_img (numpy.ndarray): 患者上传的图像
            Now 已由`Gradio`转化成了numpy数组
            input_file (str): 一个临时文件名, 指向上传的文件
            input_age (int): 患者年龄
            input_stature (float): 患者身高
            input_weight (float): 患者体重
            input_gender (str): 患者性别
            input_doctor (str): 医生意见
        Returns:
            Dict: 一系列更新Gradio组件的命令
        """
        if input_img is not None and input_file is None:
            # 仅上传图片时
            (
                ventricle_img,
                skull_img,
                diagnose_result,
                evans_index,
                ventricle_len,
                skull_len,
            ) = diagnose(
                input_img, input_age, input_stature, input_weight, input_gender
            )
            md_text = MD_TEMPLATE.format(
                name=input_name,
                age=input_age,
                stature=input_stature,
                weight=input_weight,
                gender=input_gender,
                evans_index=evans_index,
                ventricle_len=ventricle_len,
                skull_len=skull_len,
                ai_diagnosis=diagnose_result,
                doc_opinion=input_doctor,
            )
            return {
                output_panel: gr.update(visible=True),
                output_img_ventricle: ventricle_img,
                output_img_skull: skull_img,
                output_diagnosis: diagnose_result,
                report: gr.update(value=md_text),
                error_box: gr.update(value="", visible=False),
            }
        elif input_file is not None and input_img is None:
            # 仅上传文件时
            # 此处调用的函数应该存放在 ./utils/doctor.py 文件中
            # 例如对文件进行切片, 评价切片图像的好坏等
            # 代码或许类似下面这样:
            # for sliced_img in eg_func_slicer(input_file):
            #     if eg_func_evaluator(sliced_img) > 0.5:
            #         # 评价为可用的切片
            #         (
            #             ventricle_img,
            #             skull_img,
            #             diagnose_result,
            #             evans_index,
            #             ventricle_len,
            #             skull_len,
            #         ) = diagnose(
            #             sliced_img, input_age, input_stature, input_weight, input_gender
            #         )
            # 用诊断结果更新WebUI中各个组件的状态和值
            best_evans_index = 0
            best_ventricle_img = None
            best_skull_img = None
            best_diagnose_result = None
            best_ventricle_len = None
            best_skull_len = None
            for sliced_img in nii_file_slicer(input_file):
                (
                    cur_ventricle_img,
                    cur_skull_img,
                    cur_diagnose_result,
                    cur_evans_index,
                    cur_ventricle_len,
                    cur_skull_len,
                ) = diagnose(
                    sliced_img, input_age, input_stature, input_weight, input_gender
                )
                if cur_evans_index > best_evans_index:
                    best_evans_index = cur_evans_index
                    best_ventricle_img = cur_ventricle_img
                    best_skull_img = cur_skull_img
                    best_diagnose_result = cur_diagnose_result
                    best_ventricle_len = cur_ventricle_len
                    best_skull_len = cur_skull_len
            md_text = MD_TEMPLATE.format(
                name=input_name,
                age=input_age,
                stature=input_stature,
                weight=input_weight,
                gender=input_gender,
                evans_index=best_evans_index,
                ventricle_len=best_ventricle_len,
                skull_len=best_skull_len,
                ai_diagnosis=best_diagnose_result,
                doc_opinion=input_doctor,
            )
            return {
                output_panel: gr.update(visible=True),
                output_img_ventricle: best_ventricle_img,
                output_img_skull: best_skull_img,
                output_diagnosis: best_diagnose_result,
                report: gr.update(value=md_text),
                error_box: gr.update(value="", visible=False),
            }
        else:
            # 不合法的输入
            return {
                error_box: gr.update(value="请上传图片或文件二者之一", visible=True)
            }
    diagnose_btn.click(
        fn=content_wrapper,
        inputs=[
            input_name,
            input_img,
            input_file,
            input_age,
            input_stature,
            input_weight,
            input_gender,
            input_doctor,
        ],
        outputs=[
            output_img_ventricle,
            output_img_skull,
            output_diagnosis,
            error_box,
            report,
            output_panel,
        ],
    )
if __name__ == "__main__":
    demo.queue()
    demo.launch()