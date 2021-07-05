import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, 2)
    def forward(self, x, hidden):
        count = len(x)  # sequence length
        output = torch.Tensor()

        for idx in range(count):
            hidden = self.rnncell(x[:, idx], hidden)
            output = torch.cat((output, hidden))
        output = output.reshape(len(x), -1, self.hidden_size)
        output  = self.fc(output[:,-1])
        return output, hidden


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

    final_t, final_x = coordinate_array.T
    plt.plot(final_t, final_x)
    plt.grid(True)
    plt.show()

def main():

    make_sin_dataset()

if __name__ == '__main__':
    main()
