## LifanGu PyCharm+Miniconda, python3.7.x, PyTorch;
# Function: DICOM-nii.gz-tif-png-256×256JPEG renaming included; 20230913 version1.0.0(The Second Test!)
# Part 00-01: slice and Directory;
#20230914 SUCCESS!!!

import os
import pydicom
import nibabel as nib
import numpy as np
input_dir = "F:\\BEI00slice_v1.0.0\\SLICE00 DICOM STORE"
output_dir = "F:\\BEI00slice_v1.0.0\\SLICE01 NIIGZ STORE"
# 遍历DICOM文件夹
for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        output_path = os.path.join(output_dir, folder + ".nii.gz")
        # 读取DICOM文件
        dicom_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        dicom_data = [pydicom.dcmread(file) for file in dicom_files]
        # 获取像素数组
        pixel_array = [dicom.pixel_array for dicom in dicom_data]
        # 创建NIfTI文件
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(np.stack(pixel_array, axis=-1).astype(np.float32), affine)
        # 保存NIfTI文件
        nib.save(nifti_img, output_path)
# 输出成功信息
print("dicom2nii.gz successed!")

import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image
# 设置文件夹路径
input_folder = "F:\\BEI00slice_v1.0.0\\SLICE01 NIIGZ STORE"
output_folder = "F:\\BEI00slice_v1.0.0\\SLICE02 TIF STORE"
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)
# 查找所有的nii.gz文件
nii_files = glob.glob(os.path.join(input_folder, '*.nii.gz'))
for nii_file in nii_files:
    # 提取文件名和创建对应的输出文件夹
    file_name = os.path.basename(nii_file)
    folder_name = os.path.splitext(file_name)[0].replace('.nii', '_tif')
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)
    # 加载nii.gz文件
    nii_image = nib.load(nii_file)
    data = nii_image.get_fdata()
    # 对图像数据进行归一化和转换
    data = np.interp(data, (data.min(), data.max()), (0, 65535))
    data = data.astype(np.uint16)
    # 从y轴切片，并保存为tif
    slice_count = data.shape[1]
    for i in range(slice_count):
        slice_name = file_name[:-7] + '_' + str(i).zfill(3) + '.tif'  # 构建切片名称
        slice_data = data[:, i, :]
        # 创建对应输出文件路径，并保存为16位灰度tif图片
        slice_output_path = os.path.join(output_path, slice_name)
        img = Image.fromarray(slice_data, mode='I;16')
        img.save(slice_output_path)
print('nii.gz2tif successed!')

import os
import glob
import nibabel as nib
from PIL import Image
# 设置文件夹路径
input_folder = "F:\\BEI00slice_v1.0.0\\SLICE01 NIIGZ STORE"
output_folder = "F:\\BEI00slice_v1.0.0\\SLICE03 PNG STORE"
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)
# 查找所有的nii.gz文件
nii_files = glob.glob(os.path.join(input_folder, '*.nii.gz'))
for nii_file in nii_files:
    # 提取文件名和创建对应的输出文件夹
    file_name = os.path.basename(nii_file)
    folder_name = os.path.splitext(file_name)[0].replace('.nii', '_png')
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)
    # 加载nii.gz文件
    nii_image = nib.load(nii_file)
    data = nii_image.get_fdata()
    # 从y轴切片，并保存为png
    slice_count = data.shape[1]
    for i in range(slice_count):
        slice_name = file_name[:-7] + '_' + str(i).zfill(3) + '.png'  # 构建切片名称
        slice_data = data[:, i, :]
        # 创建对应输出文件路径，并保存为png图片
        slice_output_path = os.path.join(output_path, slice_name)
        img = Image.fromarray(slice_data)
        img = img.convert('L')  # 设置图片模式为L，即灰度图
        img.save(slice_output_path)
print('nii.gz2png successed!')

import os
import glob
import nibabel as nib
from PIL import Image, ImageOps
import numpy as np
# 设置文件夹路径
input_folder = "F:\\BEI00slice_v1.0.0\\SLICE01 NIIGZ STORE"
output_folder = "F:\\BEI00slice_v1.0.0\\SLICE04 JPEG STORE256"
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)
# 查找所有的nii.gz文件
nii_files = glob.glob(os.path.join(input_folder, '*.nii.gz'))
for nii_file in nii_files:
    # 提取文件名和创建对应的输出文件夹
    file_name = os.path.basename(nii_file)
    folder_name = os.path.splitext(file_name)[0].replace('.nii', '_jpeg256')
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)
    # 加载nii.gz文件
    nii_image = nib.load(nii_file)
    data = nii_image.get_fdata()
    # 从y轴切片，并保存为JPEG
    slice_count = data.shape[1]
    for i in range(slice_count):
        slice_name = file_name[:-7] + '_' + str(i).zfill(3) + '.jpg'  # 构建切片名称
        slice_data = data[:, i, :]#这里更改具体哪个轴切片；
        # Normalize slice_data if valid values exist
        if np.nanmin(slice_data) < np.nanmax(slice_data):
            slice_data = ((slice_data - np.nanmin(slice_data)) / (np.nanmax(slice_data) - np.nanmin(slice_data)) * 255).astype(np.uint8)
        else:
            slice_data = np.zeros_like(slice_data, dtype=np.uint8)
        # 在图片四周进行填充，宽高都填充至256像素
        img = Image.fromarray(slice_data)
        img = ImageOps.pad(img, (256, 256), color='black')
        # 创建对应输出文件路径，并保存为JPEG图片
        slice_output_path = os.path.join(output_path, slice_name)
        img.save(slice_output_path, format='JPEG')
print('nii.gz2jpeg succeeded!')

#LifanGu SFT, XJTU 2023 All rights reserved！#