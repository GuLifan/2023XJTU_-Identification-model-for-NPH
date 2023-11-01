###注意！！！一定要看第65行注释
from PIL import Image, ImageDraw
import json
import os
#!!!先透明背景标签再重叠！！！
# 定义标签到颜色的映射
label_to_color = {
    'green': (0, 255, 0),
    # 在这里添加更多标签和颜色
}

# 定义输入JSON文件夹和输出PNG文件夹
input_folder = "C:\\Users\\lifan\\Desktop\\json_store"  # 包含JSON文件的文件夹路径
output_folder = "C:\\Users\\lifan\\Desktop\\json2labelRGBA_store"  # 保存PNG文件的文件夹路径

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 循环处理输入文件夹中的每个JSON文件
for json_filename in os.listdir(input_folder):
    if json_filename.endswith('.json'):
        # 构建输入JSON文件的完整路径
        json_file_path = os.path.join(input_folder, json_filename)

        # 读取JSON文件内容
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

            # 获取图像尺寸和标注
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            shapes = data['shapes']

            # 创建一个空白的PNG图像，RGBA模式，背景透明
            img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))  # 0 is fully transparent
            draw = ImageDraw.Draw(img)

            # 在图像上绘制每个标注
            for shape in shapes:
                label = shape['label']
                points = shape['points']
                polygon = [(x, y) for x, y in points]

                # 使用颜色映射来确定填充颜色
                if label in label_to_color:
                    fill_color = label_to_color[label] + (128,)  # Add alpha channel (128 is semi-transparent)
                    draw.polygon(polygon, fill=fill_color)
                else:
                    print(f"警告：未知标签名： '{label}'，无法绘制标签。")

            # 构建输出PNG文件的完整路径
            output_filename = os.path.splitext(json_filename)[0] + '_label.png'
            output_path = os.path.join(output_folder, output_filename)

            # 保存PNG图像
            img.save(output_path, 'PNG')
            print(f'转换: {json_file_path} -> {output_path}')

print('转换完成！')

from PIL import Image
import os

# 输入文件夹路径和输出文件夹路径，这里没时间改了，要用的话先把所有原始的jpg粘贴到这个input_folder里面！！！——谷立帆
input_folder = "C:\\Users\\lifan\\Desktop\\json2labelRGBA_store"
output_folder = "C:\\Users\\lifan\\Desktop\\jpgmixed_store"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        jpg_path = os.path.join(input_folder, filename)
        png_path = os.path.join(input_folder, filename.replace('.jpg', '_label.png'))

        # 检查是否存在与当前JPG文件同名的PNG文件
        if os.path.exists(png_path):
            # 打开JPG文件和PNG文件
            jpg_image = Image.open(jpg_path)
            png_image = Image.open(png_path)

            # 确保PNG文件具有RGBA模式（包括透明度通道）
            if png_image.mode != 'RGBA':
                png_image = png_image.convert('RGBA')

            # 确保PNG和JPG文件具有相同的尺寸
            if jpg_image.size == png_image.size:
                # 合并PNG和JPG图像
                mixed_image = Image.alpha_composite(jpg_image.convert('RGBA'), png_image)

                # 构建输出文件路径
                output_filename = os.path.splitext(filename)[0] + '_mixed.jpg'
                output_path = os.path.join(output_folder, output_filename)

                # 保存混合后的图像为JPG
                mixed_image = mixed_image.convert('RGB')
                mixed_image.save(output_path, 'JPEG')
                print(f'Mixed: {jpg_path} + {png_path} -> {output_path}')
            else:
                print(f'Warning: Image sizes do not match for {jpg_path} and {png_path}. Skipping.')

print('Mixing complete!')
