import torch
import cv2
import os
import glob
import random
import numpy as np
from torch.utils.data import Dataset


class Img_Loader(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.imgs_path = glob.glob(os.path.join(self.root_dir, "image/*.jpg"))
 
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
        
        # 处理标签, 将像素值为255的变为1
        if label.max() > 1:
            label = label / 255
        
        # 随机增强
        option = random.choice([-1, 0, 1, 2])
        if option != 2:
            image = self.augment(image, option)
            label = self.augment(label, option)
        
        return image, label
    
    def __len__(self):
        """返回数据集的大小

        Returns:
            int: 数据集的大小
        """
        return len(self.imgs_path)
    
    
if __name__ == "__main__":
    img_lib = Img_Loader("./data/train")
    print("Loaded data counts: ", len(img_lib))
    traner_loader = torch.utils.data.DataLoader(
        dataset=img_lib,
        batch_size=1,
        shuffle=True,
    )
    
    for img, lable in traner_loader:
        print(img.shape)
