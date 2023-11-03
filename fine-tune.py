# 可调节的参数有:
# 训练层
#   epochs, batch_size, lr初始值
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for train
data_path = "./data/train"
output_path = "./output"
# for predict and eval
sources_path = glob.glob("data/train/image/*.jpg")
labels_path = glob.glob("data/train/label/*.png")
# perflog
train_perflog = []
best_record = {"avg_accuracy": 0}


def fine_tune():
    for epochs in range(50, 210, 10):
        for batch_size in range(1, 51):
            for lr in np.arange(0.01, 0.11, 0.01):
                for scale in range(1, 21):
                    for threshold in np.arange(0.1, 1.1, 0.1):
                        net = UNet(n_channels=1, n_classes=1, bilinear=True)
                        net.to(device)
                        pprint(f"Params: epochs={epochs}, batch_size={batch_size}, lr={lr}, scale={scale}, threshold={threshold}")
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
                            predict(net, device, sources_path, threshold)
                            avg_accuracy = eval(sources_path, labels_path)
                            pprint(f"Result: loss={loss}, accuracy={avg_accuracy}")                            
                        except:
                            pprint("Train failed")
                            loss = 1
                            avg_accuracy = 0
                        
                        record = {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "scale": scale,
                            "threshold": threshold,
                            "loss": loss,
                            "avg_accuracy": avg_accuracy,
                        }
                        del net
                        if record["avg_accuracy"] > best_record["avg_accuracy"]:
                            best_record = record
                        train_perflog.append(record)
                        

if __name__ == "__main__":
    fine_tune()
    json.dump(train_perflog, open("train_perflog.json", "w"), indent=4, ensure_ascii=False)
    json.dump(best_record, open("best_record.json", "w"), indent=4, ensure_ascii=False)