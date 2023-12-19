def tell_evans(ventricle_img, skull_img) -> float:
    """根据脑室识别结果和颅骨识别结果计算出evans指数

    Args:
        ventricle_img (np.array): 脑室识别结果tensor
        skull_img (np.array): 颅骨识别结果tensor

    Returns:
        float: evans指数
    """    
    # todo
    

def doctor(evans_index: float, age: int, BMI: float, gender: str) -> (bool, str):
    """根据evans指数，年龄，BMI，性别，判断脑子是否有病

    Args:
        evans_index (float): evans指数
        age (int): 年龄
        BMI (float): BMI
        gender (str): 性别

    Returns:
        (bool, str): 判断结果, 诊断意见
    """    
    # todo