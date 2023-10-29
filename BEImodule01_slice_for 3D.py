#20230427LifanGu;Model01(based on U-Net),trianing01
print('2023/04/27BEI_model01-01')
#use pip to install inbabel which will be used to handle nii or nii.gz
#工程思路：转换nii到多张png；抓取中间几层（有脑室部分）的png；灰度处理：U-Net四层；抓出颅骨与脑实质的间隙（内侧为准）和脑室；测量同层宽度；算出Evans指数
#没有写批量处理，未来改（bushi）；如果后期发现灰阶信息丢失就重写，换个文件格式——emmm，已经发现丢失了，我现在改
#@@@,三个@标志着这是我预留的修改后门，可以拓展功能
#@@@问题来了，有些图片层厚不合适&这个切片只能切三个维度的，有的四个维度的图片切不了（在改了）
#part1:格式转换与裁切
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

        # 转换为L模式
        img_fdata = img_fdata.astype(np.uint8)  # 先转换数据类型
        #@@@这里预留了x,z方向上生成切片的可能，去掉注释符号即可
        #fnamex = f.replace('.nii.gz',' -x')            #去掉nii的后缀名创建x方向2D图像文件夹
        #img_f_pathx = os.path.join(imgfile, fnamex) #创建nii对应的x方向2D图像的文件夹
        #if not os.path.exists(img_f_pathx):
        #    os.mkdir(img_f_pathx)                #新建文件夹
        
        fnamey = f.replace('.nii.gz',' _y')            #去掉nii的后缀名创建y方向2D图像文件夹
        img_f_pathy = os.path.join(imgfile, fnamey) #创建nii对应的y方向2D图像的文件夹
        if not os.path.exists(img_f_pathy):
            os.mkdir(img_f_pathy)                #新建文件夹
        
        #fnamez = f.replace('.nii.gz',' -z')            #去掉nii的后缀名创建z方向2D图像文件夹
        #img_f_pathz = os.path.join(imgfile, fnamez) #创建nii对应的z方向2D图像图像的文件夹
        #if not os.path.exists(img_f_pathz):
        #    os.mkdir(img_f_pathz)                #新建文件夹
         
        #@@@开始转换为图像，因为只需要y轴图像，因而屏蔽了x,z方向，需要时，删除注释号
        #for i in range(img_fdata.shape[0]):                      #x方向
            #silce = img_fdata[i, :, :]        
            #img_pil_slice = Image.fromarray(silce, mode='L')
            #img_pil_slice.save(os.path.join(img_f_pathx,'{}.png'.format(i))) #保存图像
        
        #@@@可以修改起始层！！！
        start_idx = 40  # 起始层为40！！！
        end_idx = img_fdata.shape[0]+32  # 终止索引为集合的最后一个索引加上32，防止需要的脑实质没切出来

        for i in range(start_idx, end_idx):  # 从起始索引到终止索引循环
            silce = img_fdata[:, i, :]         
            img_pil_slice = Image.fromarray(silce, mode='L')
            img_pil_slice.save(os.path.join(img_f_pathy,'{}.png'.format(i))) 
       
        #for i in range(img_fdata.shape[0]):                      #z方向
            #silce = img_fdata[:, :, i] 
            #img_pil_slice = Image.fromarray(silce, mode='L')
            #img_pil_slice.save(os.path.join(img_f_pathz,'{}.png'.format(i))) #保存图像
                        
if __name__ == '__main__':
    #输入输出在这改
    filepath = "F:\\BEI FM Data slice"
    imgfile = "F:\\BEI train Data85nii"
    nii_to_image(filepath,imgfile)
    print('Slice Done!切片完成！')

#3D类型nii切片完成！4D类型nii切片使用4Dmodule！