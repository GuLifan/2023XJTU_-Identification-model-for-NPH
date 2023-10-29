#20230427LifanGu;Model01(based on U-Net),trianing01
print('2023/04/27BEI_model01-01')
#use pip to install inbabel which will be used to handle nii or nii.gz
#工程思路：转换nii到多张png；抓取中间几层（有脑室部分）的png；灰度处理：U-Net四层；抓出颅骨与脑实质的间隙（内侧为准）和脑室；测量同层宽度；算出Evans指数
#没有写批量处理，未来改（bushi）；如果后期发现灰阶信息丢失就重写，换个文件格式——emmm，已经发现丢失了，我现在改
#@@@,三个@标志着这是我预留的修改后门，可以拓展功能
#@@@问题来了，有些图片层厚不合适&这个切片只能切三个维度的，四维切出来就乱码啊？？？阿巴阿巴语无伦次抽搐阴暗爬行

#两个py，现在用这个slice_for_4D保证不会被卡住。未来根据slice_for_3D再扩充功能

#part1:格式转换


import os                #用于遍历文件夹
import imageio           #图像io
from PIL import Image
import numpy as np
import nibabel as nib     #nii文件io
def nii_to_image(filepath, imgfile):
    filenames = os.listdir(filepath)  #指定nii所在的文件夹
    for f in filenames:
        #开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path,)                #读取nii
        img_fdata = img.get_fdata()

        # 判断是否为四维数据
        if len(img_fdata.shape) == 4:
            for idx in range(img_fdata.shape[3]):
                fname = f.replace('.nii.gz', f'_t{idx+1}')   #去掉nii的后缀名创建第四维文件夹
                img_f_path = os.path.join(imgfile, fname) #创建nii对应的第四维图像的文件夹
                if not os.path.exists(img_f_path):
                    os.mkdir(img_f_path)                #新建文件夹
                    
                # 开始转换为图像
                start_idx = 100  # 起始层为100！！！
                end_idx = img_fdata.shape[0]+0  # 终止索引为集合的最后一个索引加上______(自定义)，防止需要的脑实质没切出来

                for i in range(start_idx, end_idx):  # 从起始索引到终止索引循环
                    silce = img_fdata[:, :, :, idx][:, i, :]         
                    img_pil_slice = Image.fromarray(silce, mode='L')
                    img_pil_slice.save(os.path.join(img_f_path,f'{f}_t{idx+1}_z{i}.png')) 

        # 如果不是四维数据，则按照原来的方式处理
        else:
            # 转换为L模式
            img_fdata = img_fdata.astype(np.uint8)  # 先转换数据类型

            fnamey = f.replace('.nii.gz',' _y')            #去掉nii的后缀名创建y方向2D图像文件夹
            img_f_pathy = os.path.join(imgfile, fnamey) #创建nii对应的y方向2D图像的文件夹
            if not os.path.exists(img_f_pathy):
                os.mkdir(img_f_pathy)                #新建文件夹

            # 开始转换为图像
            start_idx = 100  # 起始层为100！！！
            end_idx = img_fdata.shape[0]  # 终止索引为集合的最后一个索引加上32，防止需要的脑实质没切出来

            for i in range(start_idx, end_idx):  # 从起始索引到终止索引循环
                silce = img_fdata[:, i, :]         
                img_pil_slice = Image.fromarray(silce, mode='L')
                img_pil_slice.save(os.path.join(img_f_pathy,f'{f}_z{i}.png')) 

if __name__ == '__main__':
    #输入输出在这改
    filepath = 'F:\\BEI train Data20230427_1st'
    imgfile = 'F:\\20230427model01train01saved slice'
    nii_to_image(filepath,imgfile)
    print('Slice Done!切片完成！')
