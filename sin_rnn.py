import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x, hidden):
        output, h = self.rnn(x, hidden)
        output = self.fc(output[:, -1])
        return output, h

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

def get_dataset():

    dataset = np.array([(0,0)])

    for t in range (1, 1081):

        if t<361:
            x = np.sin(np.radians(t))
            dataset = np.append(dataset, np.array([(t,x)]), axis=0)
        elif t < 721:
            x = 3*np.sin(np.radians(t))
            dataset = np.append(dataset, np.array([(t,x)]), axis=0)
        else:
            x = 5*np.sin(np.radians(t))
            dataset = np.append(dataset, np.array([(t,x)]), axis=0)

    print("dataset shape:", dataset.shape)
    print("dataset type:", type(dataset))
    return dataset

def train (data_loader, model, criterion, optimizer, epoch):

    for i in range(epoch):

      train_loss = 0
      print("epoch:", i)

      for j, (input_data, target) in enumerate(data_loader):

            model.zero_grad()

            hidden = torch.zeros(20, 20)
            output, hidden = model(input_data.float(), hidden.float())
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()

            print(j, ":", loss.item())

            train_loss += loss.item()
            ave_train_loss = train_loss/len(data_loader.dataset)
            train_loss_list.append(ave_train_loss)
            print("ave:" ,ave_train_loss)
            print()


def test(test_points, model):

    record_output = np.array([[0.0, 0.0]])

    for i in range(100):
        hidden = torch.zeros(20, 20)
        test_output, hidden = model(test_points.float(), hidden.float())
        test_points = test_points[1:20]
        test_poits.append(test_output)
        test_output = test_output.reshape(1,2).detach().numpy()
        record_output = np.concatenate(([record_output, test_output]), axis=0)
    return record_output

def drawing_loss_graph(epoch, train_loss_list):

    loss_fig = plt.figure()
    plt.plot(range(epoch), train_loss_list, linestyle='-', label='train_loss')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def main():

    batch_size  = 360 # 1つのミニバッチのデータの数
    epoch = 100  # epoch数
    time_step = batch_size

    input_size = 2
    hidden_size = 20

    points= get_dataset()
    dataset = my_dataset(points, time_step)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    """
    print("test points:", test_points.shape)
    print(test_points)
    print(type(test_points))

    test_points = torch.from_numpy(test_points.astype(np.float32)).clone()

    print("test points:", test_points.shape)
    print(type(test_points))
    """

    model = Model(input_size, hidden_size)
    criterion = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    first_point = [[0.0,0.0]]
    first_point = torch.Tensor(first_point)

    model.train()
    train(data_loader, model, criterion, optimizer, epoch)

    model.eval()
    final_test_output = test(test_points, model)

    final_x, final_y = final_test_output.T
    ims = final_test_output.T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    plt.plot(x, y, label="original", color = "y")
    plt.scatter(final_x, final_y, label="trained", color = "b")


    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.show()

if __name__ == '__main__':
    main()
