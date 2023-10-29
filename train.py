import torch.nn as nn
import torch
import os
from unet_model import UNet
from dataset import Img_Loader
from torch import optim
from pprint import pprint
from args import parser


def train_net(net, device, data_path, output_path, epochs=50, batch_size=1, lr=0.001):
    output_model = os.path.join(output_path, "unet.pth")
    
    img_dataset = Img_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset=img_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(net.parameters(), lr=lr)
    critirion = nn.BCEWithLogitsLoss() # 试试这个的效果
    best_loss = float("inf")
    
    for epoch in range(epochs):
        # 开启训练模式
        net.train()
        
        for img, label in train_loader:
            optimizer.zero_grad()
            
            # 全精度训练
            img = img.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            
            # 选择损失最小的模型保存
            predict = net(img)
            loss = critirion(predict, label)
            pprint(f"#{epoch} loop. Loss rate: ", loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), output_model)
                
            # 更新参数
            loss.backward()
            optimizer.step()
            

if __name__ == "__main__":
    arg = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=1, n_classes=1)
    net.to(device)
    
    train_net(net, device, arg.data_path, arg.output_model, arg.batch_size, arg.epochs, arg.lr)