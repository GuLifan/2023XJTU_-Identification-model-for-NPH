#20230427LifanGu;Model01(based on U-Net),trianing01
print('2023/04/27BEI_model01-01')
#这是用来将dicom转换nii.gz的程序
#不改变存储位置

import os
import pydicom
import nibabel as nib
import numpy as np

# 大文件夹的路径
root_dir = 'F:\\BEI train Data20230427_1st'

# 遍历大文件夹下所有子文件夹
for dirpath, dirnames, filenames in os.walk(root_dir):
    # 检查文件夹名称是否含有"dcm"
    if "dcm" in dirpath.lower():
        # 提取该文件夹中的所有DICOM文件
        dicom_files = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(".dcm")]
        if len(dicom_files) > 0:
            # 读取第一个DICOM文件，并获取其像素数组的维度
            img_shape = pydicom.dcmread(dicom_files[0]).pixel_array.shape
            # 检查其他DICOM文件的像素数组维度是否一致
            for dicom_file in dicom_files[1:]:
                if pydicom.dcmread(dicom_file).pixel_array.shape != img_shape:
                    raise ValueError("The pixel array shapes of DICOM files are not consistent.")
            # 将DICOM文件读入为NIfTI格式
            nifti_data = np.stack([pydicom.dcmread(dicom_file).pixel_array for dicom_file in dicom_files])
            nifti_img = nib.Nifti1Image(nifti_data, None)
            # 将NIfTI格式的图像保存为nii.gz文件
            output_file = os.path.join(dirpath, os.path.basename(dirpath) + ".nii.gz")
            nib.save(nifti_img, output_file)
            print('Transform Done!转换完毕！')
