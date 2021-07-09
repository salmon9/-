import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class my_dataset(torch.utils.data.Dataset):
    def __init__(self, points, time_step):

        self.time_step = time_step
        self.points = points
        self.len = len(points)

    def __getitem__(self, index):

        input_data = np.zeros((self.time_step, 2))
        for i in range(self.time_step):
            input_data[i] = self.points[index-self.time_step+i] + np.random.normal(scale=0.001)
        target = self.points[index] + np.random.normal(scale=0.001)

        return input_data, target

    def __len__(self):
        return self.len


def make_sin_dataset():

    coordinate_array = np.array([(0,0)])

    for t in range (1,1081):

        if t<361:
            x = np.sin(np.radians(t))
            coordinate_array = np.append(coordinate_array, np.array([(t,x)]), axis=0)
        elif t < 721:
            x = 3*np.sin(np.radians(t))
            coordinate_array = np.append(coordinate_array, np.array([(t,x)]), axis=0)
        else:
            x = 5*np.sin(np.radians(t))
            coordinate_array = np.append(coordinate_array, np.array([(t,x)]), axis=0)

    print("coordinate_array", coordinate_array.shape)
    print(coordinate_array)

    new_array = coordinate_array
    for i in range(10):
        temp_array = coordinate_array+i
        new_array = np.vstack((new_array, temp_array))

    print(new_array.shape)
    print(new_array)
    plt.plot(new_array.T)
    plt.grid(True)
    plt.show()

def main():

    make_sin_dataset()

if __name__ == '__main__':
    main()
