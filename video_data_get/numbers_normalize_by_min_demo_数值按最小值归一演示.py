import numpy as np
import random

numbers = list(range(70, 1000))  # 生成1到100的数字列表
random.shuffle(numbers)  # 打乱数字列表

print("打乱后的数字列表:", numbers)

min_value = min(numbers)

print("列表中的最小值:", min_value)

numbers_rate = [numbers//min_value for numbers in numbers]

print("每个数字除以最小值后的结果:", numbers_rate)
