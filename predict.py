import glob
import numpy as np
import torch
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from unet.unet_model import UNet
from utils.args import parser
from config.train_params import CATEGORY


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
    label = cv2.imread(label_path)
    pred = cv2.imread(pred_path)

    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

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
    predict(net, device, sources_path)
    print("avg_accuracy", eval(sources_path, labels_path))
