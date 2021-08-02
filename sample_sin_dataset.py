import numpy as np
import matplotlib.pyplot as plt
import torch
from random import uniform


def make_data(num_div, cycles, offset=0):
    step = 2 * np.pi / num_div
    res0 = [i*np.sin(step * i + offset) for i in range(num_div * cycles + 1)]
    res1 = [i*np.sin(step * i + offset) + uniform(-0.02, 0.02) for i in range(num_div * cycles + 1)]

    print(type(res1))

    return res0, res1

def main():

    num_div = 100  # 1周期の分割数
    cycles = 4  # 周期数
    num_batch = 25  # 1つの時系列データのデータ数
    sample_data, _ = make_data(num_div, cycles)

    plt.plot(sample_data)
    plt.grid()

    plt.show()

if __name__ == '__main__':
    main()
