# 可调节的参数有:
# 训练层
#   bilinear, epochs, batch_size, lr初始值
# 数据处理层
#   scale
# 预测结果
#   threshold
# 评价指标: predict.eval()返回的准确值

from train import train_net
from predict import predict, eval
from pprint import pprint
from unet.unet_model import UNet
import numpy as np
import torch
import glob
import json


# shared
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for train
data_path = "./data/ventricle/train"
output_path = "./output"
# for predict and eval
sources_path = glob.glob("data/ventricle/train/image/*.jpg")
labels_path = glob.glob("data/ventricle/train/label/*.png")


def fine_tune():
    # perflog
    max_batch = 51
    train_perflog = []
    best_record = {"avg_accuracy": 0}
    for epochs in range(100, 310, 10):
        for scale in range(1, 21):
            for lr in np.arange(0.0001, 0.011, 0.0004):
                for batch_size in range(1, max_batch):
                    net = UNet(n_channels=1, n_classes=1, bilinear=False)
                    net.to(device)
                    pprint(
                        f"Train params: epochs={epochs}, batch_size={batch_size}, lr={lr}, scale={scale}"
                    )
                    try:
                        loss = train_net(
                            net,
                            device,
                            data_path,
                            output_path,
                            epochs,
                            batch_size,
                            lr,
                            scale,
                        )
                    except:
                        pprint("Train failed")
                        # train失败了大概率是因为爆显存了, 此处调小显存, 直接开始新一轮循环
                        max_batch = batch_size - 1
                        break
                    for threshold in np.arange(0.1, 2.1, 0.1):
                        pprint(f"|----Predict param: threshold={threshold}")
                        try:
                            predict(net, device, sources_path, threshold)
                            avg_accuracy = eval(sources_path, labels_path)
                            pprint(
                                f"|    |----Result: loss={loss}, accuracy={avg_accuracy}"
                            )
                        except:
                            pprint("|    |----Predict failed")
                            loss = 1
                            avg_accuracy = 0

                        record = {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "scale": scale,
                            "threshold": threshold,
                            "loss": loss.item(),
                            "avg_accuracy": avg_accuracy,
                        }
                        if record["avg_accuracy"] > best_record["avg_accuracy"]:
                            best_record = record
                        train_perflog.append(record)
                    # del net

                    json.dump(
                        train_perflog, open("train_perflog.json", "w"), indent=4, ensure_ascii=False
                    )
                    json.dump(best_record, open("best_record.json", "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fine_tune()
