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
        self.hidden_size = hidden_size
    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = torch.zeros(x.shape[0], hidden_size)
        output, h = self.rnn(x, hidden)
        output = self.fc(output)
        return output, h

class my_dataset(torch.utils.data.Dataset):
    def __init__(self, points):

        self.points = points
        self.len = len(points)

    def __getitem__(self, index):

        input_data = self.points[index] + np.random.normal(scale=0.001)
        target = self.points[index]

        return input_data, target

    def __len__(self):
        return self.len

def get_dataset():

    dataset = np.array([[[0]]])

    for t in range (1, 1081):

        if t<361:
            x = np.sin(np.radians(t))
            dataset = np.append(dataset, np.array([[[x]]]), axis=0)
        elif t < 721:
            x = 3*np.sin(np.radians(t))
            dataset = np.append(dataset, np.array([[[x]]]), axis=0)
        else:
            x = 5*np.sin(np.radians(t))
            dataset = np.append(dataset, np.array([[[x]]]), axis=0)

    print("dataset shape:", dataset.shape)
    print("dataset type:", type(dataset))
    print(dataset)

    return dataset

def dataset_mizumashi(dataset, seq_num):

    new_dataset = dataset
    for i in range(1,seq_num):
        temp_array = dataset+i
        new_dataset = np.append(new_dataset, temp_array, axis=2)

    print("dataset_mizumashi")
    print(new_dataset.shape)
    new_dataset = np.transpose(new_dataset, (2, 0, 1))
    print(new_dataset.shape)
    return new_dataset

def train (data_loader, model, criterion, optimizer, epoch):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loss_list = []
    for i in range(epoch):
        train_loss = 0
        print("epoch:", i+1)

        for j, (input_data, target) in enumerate(data_loader):

            model.zero_grad()

            hidden = torch.zeros(1, 1080, 5)
            output, hidden = model(input_data[:,:-1].float(), hidden.float())
            loss = criterion(output, target[:,1:].float())
            loss.backward()
            optimizer.step()

            print(j+1, ":", loss.item())

            train_loss += loss.item()
        ave_train_loss = train_loss/len(data_loader.dataset)
        train_loss_list.append(ave_train_loss)
        print("ave:" ,ave_train_loss)
        print()


def test(test_points, model):

    record_output = np.array([0])

    for i in range(1080):
        hidden = torch.zeros(1, 1080, 5)
        test_output, hidden = model(input_data[:,:-1].float(), hidden.float())
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

    batch_size  = 10 # 1つのミニバッチのデータの数
    epoch = 100  # epoch数

    input_size = 1
    hidden_size = 5

    seq_num = 100

    points= get_dataset()

    points = dataset_mizumashi(points, seq_num)

    print("points")
    print(points.shape)
    dataset = my_dataset(points)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    print("dataloader type", type(data_loader))
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

    first_point = [0]
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
