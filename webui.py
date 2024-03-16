import gradio as gr
import io
from PIL import Image
import torch
import numpy as np
from unet.unet_model import UNet
from utils.doctor import tell_evans, doctor, nii_file_slicer

try:
    from config.train_params import THRESHOLD
except:
    THRESHOLD = 0.5

# 使用指南
GUIDELINE = """
使用步骤:

1. 上传患者颅内影像图片, 或者是患者颅内影像文件(.nii文件). 注意: **只需要上传二者之一即可**
2. 填写患者信息, 包括姓名, 年龄, 身高, 体重, 性别等
3. 填写医生诊断意见
4. 点击 **开始诊断** 按钮

诊断结果和细节将会出现在按钮下方
"""

# 诊断报告模板
MD_TEMPLATE = """
# 诊断报告

## 患者信息

- 患者姓名: {name}
- 患者年龄: {age}
- 患者身高: {stature}
- 患者体重: {weight}
- 患者性别: {gender}

## 诊断结果

- EI指数: {evans_index}
- 脑室宽度: {ventricle_len}
- 颅骨宽度: {skull_len}

## 诊断意见

- AI意见: {ai_diagnosis}
- 医生意见: {doc_opinion}
"""

# 加载本地模型
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


def diagnose(input_img, input_age, input_stature, input_weight, input_gender):
    """根据输入的图像, 年龄, 身高, 体重, 性别等进行诊断

    Args:
        input_img (numpy.array): 输入的灰度图像数组
        input_age (int): 输入的年龄
        input_stature (float): 输入的身高
        input_weight (float): 输入的体重
        input_gender (str): 输入的性别

    Returns:
        Image: 脑室识别图像
        Image: 颅骨识别图像
        str: 诊断意见
        float: Evans指数
        float: 脑室长度
        float: 颅骨长度
    """
    original_image = Image.fromarray(input_img)
    img = input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    # 识别脑室
    ventricle_img = ventricle_unet(img_tensor)
    ventricle_img = np.array(ventricle_img.data.cpu()[0])[0]
    ventricle_img[ventricle_img >= THRESHOLD] = 255
    ventricle_img[ventricle_img < THRESHOLD] = 0
    # print("******预测成功******")
    ventricle_image = Image.fromarray(ventricle_img)
    ventricle_image = ventricle_image.convert("RGB")

    # 识别颅骨
    skull_img = skull_unet(img_tensor)
    skull_img = np.array(skull_img.data.cpu()[0])[0]
    skull_img[skull_img >= THRESHOLD] = 255
    skull_img[skull_img < THRESHOLD] = 0
    skull_image = Image.fromarray(skull_img)
    skull_image = skull_image.convert("RGB")

    # 完成诊断函数后取消下面这段的注释
    evans_index, ventricle_len, skull_len = tell_evans(ventricle_img, skull_img)
    ill, diagnosis = doctor(
        evans_index, input_age, input_stature, input_weight, input_gender
    )
    print(f"***evans: {evans_index}")
    indicator = (255, 0, 0) if ill else (0, 255, 0)  # 绿色健康, 红色有病

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
    # print("******染色成功******")
    # 叠加
    ventricle_image.paste(original_image, (0, 0), original_image)
    skull_image.paste(original_image, (0, 0), original_image)
    # print("******叠加成功******")
    return (
        ventricle_image,
        skull_image,
        diagnosis,
        evans_index,
        ventricle_len,
        skull_len,
    )


# 以下是gradio webui组件
guideline = gr.Markdown(value=GUIDELINE, label="使用指南")
input_img = gr.Image(
    image_mode="L",
    sources=["upload"],
    type="numpy",
    label="患者颅内影像",
)
input_file = gr.File(label="患者颅内影像文件")
input_name = gr.Textbox(value="示例", label="患者姓名", interactive=True)
input_age = gr.Number(
    minimum=0, maximum=100, value=20, label="患者年龄", interactive=True
)
input_stature = gr.Number(
    minimum=0, maximum=250, value=170, label="患者身高(cm)", interactive=True
)
input_weight = gr.Number(
    minimum=0, maximum=200, value=60, label="患者体重(kg)", interactive=True
)
input_gender = gr.Dropdown(
    choices=["男性", "女性"], value="男性", label="患者性别", interactive=True
)
input_doctor = gr.Textbox(label="医生意见", value="元芳, 你怎么看?", interactive=True)

output_img_ventricle = gr.Image(
    type="pil",
    image_mode="RGB",
    label="脑室识别结果",
    show_label=True,
    show_share_button=True,
)
output_img_skull = gr.Image(
    type="pil",
    image_mode="RGB",
    label="颅骨识别结果",
    show_label=True,
    show_share_button=True,
)
output_diagnosis = gr.Textbox(value="哼哼哼, 啊啊啊啊啊啊啊啊!!!", label="诊断意见")
report = gr.Markdown(value="诊断报告", label="诊断报告")

error_box = gr.Textbox(label="出错了!", visible=False)
diagnose_btn = gr.Button(value="开始诊断", size="lg")


with gr.Blocks(title="UNet颅内影像识别") as demo:
    guideline.render()
    with gr.Column() as input_panel:
        with gr.Row() as input_sources:
            input_img.render()
            input_file.render()
            with gr.Column() as input_details:
                input_name.render()
                input_age.render()
                input_stature.render()
                input_weight.render()
                input_gender.render()
        input_doctor.render()
        diagnose_btn.render()
    error_box.render()
    with gr.Column(visible=False) as output_panel:
        output_diagnosis.render()
        with gr.Row() as output_alts:
            with gr.Accordion(label="诊断报告") as output_report:
                report.render()
            with gr.Accordion(label="区域识别") as output_images:
                with gr.Row():
                    output_img_ventricle.render()
                    output_img_skull.render()

    def content_wrapper(
        input_name,
        input_img,
        input_file,
        input_age,
        input_stature,
        input_weight,
        input_gender,
        input_doctor,
    ):
        """这是负责处理输入和包装输出的函数, 在调用模型进行推理前, 需要判断输入是否合法, 以及根据输入类型(图片/文件)来选择处理方法; 在调用模型推理后, 需要根据模型结果更新WebUI中各个组件的状态和值

        Args:
            input_name (str): 患者姓名
            input_img (numpy.ndarray): 患者上传的图像, 已由`Gradio`转化成了numpy数组
            input_file (str): 一个临时文件名, 指向上传的文件
            input_age (int): 患者年龄
            input_stature (float): 患者身高
            input_weight (float): 患者体重
            input_gender (str): 患者性别
            input_doctor (str): 医生意见

        Returns:
            Dict: 一系列更新Gradio组件的命令
        """
        if input_img is not None and input_file is None:
            # 仅上传图片时
            (
                ventricle_img,
                skull_img,
                diagnose_result,
                evans_index,
                ventricle_len,
                skull_len,
            ) = diagnose(
                input_img, input_age, input_stature, input_weight, input_gender
            )

            md_text = MD_TEMPLATE.format(
                name=input_name,
                age=input_age,
                stature=input_stature,
                weight=input_weight,
                gender=input_gender,
                evans_index=evans_index,
                ventricle_len=ventricle_len,
                skull_len=skull_len,
                ai_diagnosis=diagnose_result,
                doc_opinion=input_doctor,
            )

            return {
                output_panel: gr.update(visible=True),
                output_img_ventricle: ventricle_img,
                output_img_skull: skull_img,
                output_diagnosis: diagnose_result,
                report: gr.update(value=md_text),
                error_box: gr.update(value="", visible=False),
            }
        elif input_file is not None and input_img is None:
            # 仅上传文件时
            # 此处调用的函数应该存放在 ./utils/doctor.py 文件中
            # 例如对文件进行切片, 评价切片图像的好坏等
            # 代码或许类似下面这样:
            # for sliced_img in eg_func_slicer(input_file):
            #     if eg_func_evaluator(sliced_img) > 0.5:
            #         # 评价为可用的切片
            #         (
            #             ventricle_img,
            #             skull_img,
            #             diagnose_result,
            #             evans_index,
            #             ventricle_len,
            #             skull_len,
            #         ) = diagnose(
            #             sliced_img, input_age, input_stature, input_weight, input_gender
            #         )
            # 用诊断结果做些什么
            best_evans_index = 0
            best_ventricle_img = None
            best_skull_img = None
            best_diagnose_result = None
            best_ventricle_len = None
            best_skull_len = None
            for sliced_img in nii_file_slicer(input_file):
                (
                    cur_ventricle_img,
                    cur_skull_img,
                    cur_diagnose_result,
                    cur_evans_index,
                    cur_ventricle_len,
                    cur_skull_len,
                ) = diagnose(
                    sliced_img, input_age, input_stature, input_weight, input_gender
                )
                if cur_evans_index > best_evans_index:
                    best_evans_index = cur_evans_index
                    best_ventricle_img = cur_ventricle_img
                    best_skull_img = cur_skull_img
                    best_diagnose_result = cur_diagnose_result
                    best_ventricle_len = cur_ventricle_len
                    best_skull_len = cur_skull_len

            md_text = MD_TEMPLATE.format(
                name=input_name,
                age=input_age,
                stature=input_stature,
                weight=input_weight,
                gender=input_gender,
                evans_index=best_evans_index,
                ventricle_len=best_ventricle_len,
                skull_len=best_skull_len,
                ai_diagnosis=best_diagnose_result,
                doc_opinion=input_doctor,
            )

            return {
                output_panel: gr.update(visible=True),
                output_img_ventricle: best_ventricle_img,
                output_img_skull: best_skull_img,
                output_diagnosis: best_diagnose_result,
                report: gr.update(value=md_text),
                error_box: gr.update(value="", visible=False),
            }

        else:
            # 不合法的输入
            return {
                error_box: gr.update(value="请上传图片或文件二者之一", visible=True)
            }

    diagnose_btn.click(
        fn=content_wrapper,
        inputs=[
            input_name,
            input_img,
            input_file,
            input_age,
            input_stature,
            input_weight,
            input_gender,
            input_doctor,
        ],
        outputs=[
            output_img_ventricle,
            output_img_skull,
            output_diagnosis,
            error_box,
            report,
            output_panel,
        ],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
