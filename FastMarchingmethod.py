#使用FastMarching算法分割（来自腾讯云LifanGu2023/05/06）
#1.首先使用各向异性扩散方法对输入图像进行平滑处理；
#2.其次对平滑后的图像进行梯度计算，生成边缘图像，在梯度计算过程中可调节高斯sigma参数，来控制水平集减速到接近边缘；
#3.然后使用逻辑回归（Sigmoid）函数对边缘图像进行线性变换，保证边界接区域近零，平坦区域接近1，回归可调参数有alpha和beta；
#4.接着手动设置置FastMarching算法的初始种子点和起始值，该种子点是水平集的起始位置。FastMarching的输出是时间跨度图，表示传播的水平集面到达的时间；
#5.最后通过阈值方法将FastMarching结果限制在水平集面传播区域而形成分割的区域。

from __future__ import print_function#可以删掉，删了就不支持python2.x
import SimpleITK as sitk
import sys
import os

if len(sys.argv) < 10:
    print("Usage: {} <inputImage> <outputImage> <seedX> <seedY> <Sigma> <SigmoidAlpha> <SigmoidBeta> <TimeThreshold>".format(sys.argv[0]))
    sys.exit(1)

inputFilename = sys.argv[1]
outputFilename = sys.argv[2]
seedPosition = (int(sys.argv[3]), int(sys.argv[4]))
sigma = float(sys.argv[5])
alpha = float(sys.argv[6])
beta = float(sys.argv[7])
timeThreshold = float(sys.argv[8])
stoppingTime = float(sys.argv[9])

# 读取输入图像
inputImage = sitk.ReadImage(inputFilename, sitk.sitkFloat32)

# 进行曲率各向异性扩散滤波
smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
smoothing.SetTimeStep(0.125)
smoothing.SetNumberOfIterations(5)
smoothing.SetConductanceParameter(9.0)
smoothingOutput = smoothing.Execute(inputImage)

# 计算梯度幅值
gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
gradientMagnitude.SetSigma(sigma)
gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

# 进行Sigmoid变换
sigmoid = sitk.SigmoidImageFilter()
sigmoid.SetOutputMinimum(0.0)
sigmoid.SetOutputMaximum(1.0)
sigmoid.SetAlpha(alpha)
sigmoid.SetBeta(beta)
sigmoid.DebugOn()
sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)

# 使用快速行进算法进行分割
fastMarching = sitk.FastMarchingImageFilter()
seedValue = 0
trialPoint = (seedPosition[0], seedPosition[1], seedValue)
fastMarching.AddTrialPoint(trialPoint)
fastMarching.SetStoppingValue(stoppingTime)
fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

# 二值化分割结果
thresholder = sitk.BinaryThresholdImageFilter()
thresholder.SetLowerThreshold(0.0)
thresholder.SetUpperThreshold(timeThreshold)
thresholder.SetOutsideValue(0)
thresholder.SetInsideValue(255)
result = thresholder.Execute(fastMarchingOutput)

# 写入输出图像
sitk.WriteImage(result, outputFilename)
print('Done!完成！')

#这是一个基于FastMarching的算法，用于分割医学图像。以下是程序各部分的功能和意义：
#使用SimpleITK库的import语句：导入SimpleITK库，该库包含用于医学图像处理的工具。
#检查输入参数：检查在命令行中是否提供了足够的参数来运行程序。如果没有提供，将显示程序的用法并退出。
#定义输入和输出文件名：使用硬编码的方式定义输入和输出文件名。这些文件名指定了将要读取和写入数据的文件。
#定义种子点位置：从命令行参数中获取种子点的位置，并将其存储在一个元组中。
#定义滤波参数：使用从命令行参数中获取的值，定义高斯滤波的标准差、Sigmoid函数的Alpha参数、Sigmoid函数的Beta参数、时间阈值和停止时间。这些参数将用于后续的滤波和分割操作。
#读取输入图像：使用SimpleITK库的ReadImage函数读取输入图像。
#执行曲率各向异性扩散滤波：使用SimpleITK库的CurvatureAnisotropicDiffusionImageFilter函数对输入图像进行滤波操作，以减少噪声和平滑图像。
#计算梯度幅值：使用SimpleITK库的GradientMagnitudeRecursiveGaussianImageFilter函数计算滤波后图像的梯度幅值。
#进行Sigmoid变换：使用SimpleITK库的SigmoidImageFilter函数对梯度幅值进行Sigmoid变换，以增强前景和背景之间的差异。
#进行快速行进算法分割：使用SimpleITK库的FastMarchingImageFilter函数，将种子点添加到算法中，并设置停止时间。然后执行快速行进算法，以生成分割结果。
#二值化分割结果：使用SimpleITK库的BinaryThresholdImageFilter函数，将分割结果二值化，并设置阈值，以生成最终的分割图像。
#写入输出图像：使用SimpleITK库的WriteImage函数，将分割图像写入到输出文件中。
#显示操作完成：在命令行中打印“Done!完成！”，表示程序成功完成。


#@@@声明：LifanGU Based on FastMarching method，2023/05/08 17:21(UTC+8:00), Version 0.1.1@@@#
#复制粘贴冒号后的内容到终端：python "F:\\Microsoft Visual Studio Code LifanGu Data\\Project BEI\\module02_distinguishing\\FastMarchingmethod.py" "F:\\FastMarchingData\\163.png" "F:\\FastMarchingResultv0.1.1\\16300.png" 72 100 0.5 -0.3 2.0 200 210
#一般sigma=0.5，alpha=-0.3，beta=2.0， timeThreshold=200， stoppingTime=210，seedPosition为期望分割区域内任意点坐标即可
#"F:\\FastMarchingData\\163.png"
#"F:\\FastMarchingResultv0.1.1\\16300.png"
#@@@这个程序的文件夹名字不要有空格！！！@@@#

#如果增加高斯滤波的标准差，则平滑效果会增强，但可能会导致细节信息损失；
#如果减小Sigmoid函数的Alpha参数，则Sigmoid函数将更慢地向输出最小值（0.0）靠近，从而使分割结果更具有鲁棒性，但可能会导致分割边界比较模糊；
#如果增大时间阈值，则算法将更倾向于将较暗的区域视为背景，而将较亮的区域视为前景，从而降低图像的灰度动态范围；
#如果减小停止时间，则算法将更快地扩散到远离选择点的区域，但也可能会将一些无关区域纳入分割结果中。
