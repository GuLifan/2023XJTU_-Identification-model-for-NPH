import argparse


parser = argparse.ArgumentParser(description="训练模型相关参数")

parser.add_argument("--batch-size", type=int, default=1, help="训练时batch_size")
parser.add_argument("--epochs", type=int, default=50, help="训练时epochs")
parser.add_argument("--lr", type=float, default=0.001, help="学习率")
parser.add_argument("--data-path", type=str, default="./data/train", help="数据集路径")
parser.add_argument("--output-path", type=str, default="./output", help="模型输出路径")