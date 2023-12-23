# 2023XJTU_-Identification-model-for-NPH

<p align="center" style="font-size: 1.5em;">UNet模型现在已经可用了!🎉</p>

<p style="color: red">本项目还在开发阶段, 目前还需实现evans指数测量

## 快速上手

1. 下载本仓库并切换到`UNet`分支
2. 安装依赖:

    ```shell
    pip install -r requirements.txt
    ```

3. 从[这里](https://github.com/Orion-zhen/project-brain/releases)下载模型文件(`unet-ventricle.pth`和`unet-skull.pth`)
4. 把下载下来的模型文件放入仓库目录中的`./output`目录
5. 运行命令:

    ```shell
    python webui.py
    ```

6. 浏览器打开网址[127.0.0.1:7860](http://127.0.0.1:7860)
7. 开始体验!🤗

## 训练模型

仓库里内置了649张打好标签的数据, 你可以直接开始训练!😃

1. 复制训练参数配置文件:

    ```shell
    cp ./config/train_params.py.example ./config/train_params.py
    ```

2. 在`./config/train_params.py`中调整训练参数, 其中`CATEGORY`的值可以取`ventricle`(代表脑室识别)或`skull`(代表颅骨识别)
3. 开始训练:

    ```shell
    python train.py
    ```

    你可以运行`python train.py --help`查看更多可选命令行参数. 训练结果默认保存在`./output`文件夹内
4. 测试训练结果:

    ```shell
    python predict.py
    ```

## developer's message

关于evans指数的测量已经由[@hbx](https://github.com/root-hbx)完成了, 你们可以增加基于evans指数和体重, 年龄, BMI等的诊断报告😘
