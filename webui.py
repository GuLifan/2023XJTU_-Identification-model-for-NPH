import gradio as gr
import io
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet.unet_model import UNet
from config.train_params import THRESHOLD
from utils.doctor import tell_evans, doctor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ventricle_unet = UNet(n_channels=1, n_classes=1)
ventricle_unet.to(device=device)
ventricle_unet.load_state_dict(
    torch.load(f"./output/unet-ventricle.pth", map_location=device)
)
ventricle_unet.eval()

skull_unet = UNet(n_channels=1, n_classes=1)
skull_unet.to(device=device)
skull_unet.load_state_dict(torch.load(f"./output/unet-skull.pth", map_location=device))
skull_unet.eval()


def visualize_percentage(percentage):
    # 创建一个新的Matplotlib图形
    _, ax = plt.subplots()

    # 创建一个水平条形图
    ax.barh(0, percentage, color="green" if percentage < 0.5 else "red")

    # 设置图形的标题和标签
    ax.set_title("Percentage Visualization")
    ax.set_xlabel("Percentage")
    ax.set_ylabel("")

    # 隐藏坐标轴
    ax.axis("off")

    # 将图形对象转换为字节流
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return buf.read()


def tell_diagnosis(percentage):
    return "这里可以让医学生根据预测结果写一段诊断报告"


def tell_ill(ventricle_img, skull_img) -> float:
    feature_img = ventricle_img + skull_img
    # ill_probability = doctor_net(feature_img)
    return 0.8


def diagnose(input_img, input_age, input_BMI, input_gender):
    original_image = Image.fromarray(input_img)
    img = input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    # 识别脑室
    ventricle_img = ventricle_unet(img_tensor)
    ventricle_img = np.array(ventricle_img.data.cpu()[0])[0]
    ventricle_img[ventricle_img >= THRESHOLD] = 255
    ventricle_img[ventricle_img < THRESHOLD] = 0
    print("******预测成功******")
    ventricle_image = Image.fromarray(ventricle_img)
    ventricle_image = ventricle_image.convert("RGB")

    # 识别颅骨
    skull_img = skull_unet(img_tensor)
    skull_img = np.array(skull_img.data.cpu()[0])[0]
    skull_img[skull_img >= THRESHOLD] = 255
    skull_img[skull_img < THRESHOLD] = 0
    skull_image = Image.fromarray(skull_img)
    skull_image = skull_image.convert("RGB")

    # 完成诊断函数后把下面这段删掉
    ill_percent = tell_ill(ventricle_img, ventricle_img)
    diagnosis = tell_diagnosis(ill_percent)
    indicator = (0, 255, 0) if ill_percent < 0.5 else (255, 0, 0)  # 绿色健康, 红色有病
    
    # 完成诊断函数后取消下面这段的注释
    # evans_index = tell_evans(ventricle_img, skull_img)
    # ill, diagnosis = doctor(evans_index, input_age, input_BMI, input_gender)
    # indicator = (255, 0, 0) if ill else (0, 255, 0) # 绿色健康, 红色有病

    # 将输出图像染色
    width, height = ventricle_image.size
    # 遍历图像的每个像素
    # ventricle
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = ventricle_image.getpixel((x, y))
            # 如果像素是黑色，则将其改为指示色
            if r == 0 and g == 0 and b == 0:
                ventricle_image.putpixel((x, y), indicator)
    # skull            
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = skull_image.getpixel((x, y))
            # 如果像素是黑色，则将其改为指示色
            if r == 0 and g == 0 and b == 0:
                skull_image.putpixel((x, y), indicator)
    print("******染色成功******")
    # 叠加
    ventricle_image.paste(original_image, (0, 0), original_image)
    skull_image.paste(original_image, (0, 0), original_image)
    print("******叠加成功******")
    return ventricle_image, skull_image, diagnosis


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    input_img = gr.Image(
                        image_mode="L",
                        sources=["upload"],
                        type="numpy",
                        width=256,
                        height=256,
                    )
                    with gr.Column():
                        input_age = gr.Number(
                            minimum=0, maximum=100, value=20, label="患者年龄"
                        )
                        input_BMI = gr.Number(
                            minimum=0, maximum=100, value=20, label="患者BMI"
                        )
                        input_gender = gr.Dropdown(
                            choices=["Male", "Female"], value="Male", label="患者性别"
                        )
            diagnose_btn = gr.Button(value="开始诊断", size="lg")

        with gr.Column():
            with gr.Accordion(label="特征提取"):
                with gr.Row():
                    output_img1 = gr.Image(
                        type="pil",
                        image_mode="RGB",
                        label="脑室",
                        show_label=True,
                        width=256,
                        height=256,
                    )
                    output_img2 = gr.Image(
                        type="pil",
                        image_mode="RGB",
                        label="颅骨",
                        show_label=True,
                        width=256,
                        height=256,
                    )
            # visualizer = gr.Image(label="患病概率", show_label=True)
            diagnosis = gr.Textbox(value="我也不知道是啥结果, 脑壳子的数据还没来", label="诊断意见")

    diagnose_btn.click(
        fn=diagnose,
        inputs=[input_img, input_age, input_BMI, input_gender],
        outputs=[output_img1, output_img2, diagnosis],
    )


if __name__ == "__main__":
    demo.launch()
