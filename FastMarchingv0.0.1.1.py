#对于同一文件下的文件的批量处理LifanGu 20230520 Version 0.0.1(beta)
#处理tiff十六位色深，应该没有灰度丢失
from __future__ import print_function
import SimpleITK as sitk
import sys
import os

inputDirectory = "F:\\FastMarchingDataTraining2"
outputDirectory = "F:\\FastMarchingResult20230521"

sigma = 0.5
alpha = -0.3
beta = 2.0

# Get a list of all PNG files in the input directory
inputFiles = [f for f in os.listdir(inputDirectory) if f.endswith('.png')]
#
#处理所有的同文件夹下的tiff
# Loop through input files
for inputFile in inputFiles:
    inputFilename = os.path.join(inputDirectory, inputFile)

    # Read the input image
    inputImage = sitk.ReadImage(inputFilename, sitk.sitkFloat32)
    # Perform curvature anisotropic diffusion filtering
    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(0.125)
    smoothing.SetNumberOfIterations(5)
    smoothing.SetConductanceParameter(9.0)
    smoothingOutput = smoothing.Execute(inputImage)
    # Calculate gradient magnitude
    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(sigma)
    gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)
    # Perform Sigmoid transformation
    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)

    # Loop through timeThreshold values
    for timeThreshold in range(90, 121, 5):
        # Loop through stoppingTime values
        for stoppingTime in range(190, 231, 10):
            # Loop through the seed positions
            for i in range(50, 91, 5):
                for j in range(70, 131, 5):
                    seedPosition = (i, j)
                    outputFilename = os.path.join(outputDirectory, "{}_{:02d}_{:02d}x{:02d}_tt{}_st{}.png".format(inputFile[:-4], i // 10, i, j, timeThreshold, stoppingTime))

                    # Use the FastMarching algorithm for segmentation
                    fastMarching = sitk.FastMarchingImageFilter()
                    seedValue = 0
                    trialPoint = (seedPosition[0], seedPosition[1], seedValue)
                    fastMarching.AddTrialPoint(trialPoint)
                    fastMarching.SetStoppingValue(stoppingTime)
                    fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

                    # Threshold the segmentation result
                    thresholder = sitk.BinaryThresholdImageFilter()
                    thresholder.SetLowerThreshold(0.0)
                    thresholder.SetUpperThreshold(timeThreshold)
                    thresholder.SetOutsideValue(0)
                    thresholder.SetInsideValue(255)
                    result = thresholder.Execute(fastMarchingOutput)

                    # Write the output image
                    sitk.WriteImage(result, outputFilename)
                    print("Saved image: ", outputFilename)

print('Done!')
#第一位0说明健康，1说明有脑积水；2~3两位是本集合的编号，后三位是图片原名