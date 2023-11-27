import gradio as gr
import cv2
from PIL import Image
import torch
import numpy as np
from unet.unet_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(n_channels=1, n_classes=1)
net.to(device=device)
net.load_state_dict(torch.load(f"./output/unet-ventricle.pth", map_location=device))
net.eval()


def predict(input_img, input_age, input_BMI, input_gender):
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    print(input_img.shape)
    img = input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    pred = net(img_tensor)
    pred = np.array(pred.data.cpu()[0])[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    print(pred.shape)
    pil_image = Image.fromarray(pred)
    original_image = Image.fromarray(input_img)
    
    output_image = pil_image.convert("RGB")
    width, height = output_image.size

    # 遍历图像的每个像素
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = output_image.getpixel((x, y))

            # 如果像素是黑色，则将其改为绿色
            if r == 0 and g == 0 and b == 0:
                output_image.putpixel((x, y), (0, 255, 0))
    
    output_image.paste(original_image, (0, 0), original_image)
    # print(output_image.shape)
    return output_image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                image_mode="L", sources="upload", type="numpy", 
                # width=256, height=256
            )
            input_age = gr.Number(minimum=0, maximum=100, value=20, label="Age")
            input_BMI = gr.Number(minimum=0, maximum=100, value=20, label="BMI")
            input_gender = gr.Dropdown(
                choices=["Male", "Female"], value="Male", label="Gender"
            )
        with gr.Column():
            with gr.Row():
                output_img = gr.Image(type="pil", image_mode="RGB", label="ventricle", show_label=True, width=256, height=256)
                output_img2 = gr.Image(type="pil", image_mode="RGB", label="skull", show_label=True, width=256, height=256)
            gr.Textbox(value="我也不知道是啥结果, 脑壳子的数据还没来", label="诊断结果")

    pred_btn = gr.Button()
    pred_btn.click(fn=predict, inputs=[input_img, input_age, input_BMI, input_gender], outputs=output_img)


if __name__ == "__main__":
    demo.launch()
