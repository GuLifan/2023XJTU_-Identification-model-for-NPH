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
import argparse


parser = argparse.ArgumentParser(description="训练模型相关参数")

parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="训练时batch_size")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练时epochs")
parser.add_argument("--lr", type=float, default=LR, help="学习率")
parser.add_argument("--data-path", type=str, default=DATA_PATH, help="数据集路径")
parser.add_argument("--output-path", type=str, default=OUTPUT_PATH, help="模型输出路径")
parser.add_argument("--scale", type=int, default=SCALE, help="图片像素值缩放大小")
parser.add_argument("--threshold", type=int, default=THRESHOLD, help="模型判别阈值")
parser.add_argument("--model-type", type=str, default=MODEL_TYPE, help="模型类型，可选unet、resnet18等")
parser.add_argument("--category", type=str, default=CATEGORY, help="模型类别，可选ventricle、skull等")
