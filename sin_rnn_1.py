import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from random import uniform


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x, hidden):
        output, h = self.rnn(x, hidden)
        output = self.fc(output[:, -1])
        return output, h


def make_data(num_div, cycles, offset=0):
    step = 2 * np.pi / num_div
    x_0 = [np.sin(step * i + offset) for i in range(num_div * cycles + 1)]
    x_1 = [np.sin(step * i + offset) + uniform(-0.02, 0.02) for i in range(num_div * cycles + 1)]

    print(type(x_0))

    return x_0, x_1

def make_train_data(num_div, cycles, num_batch, offset=0):
    x, x_w_noise = make_data(num_div, cycles, offset)
    count = len(x) - num_batch
    data = [x_w_noise[idx:idx+num_batch] for idx in range(count)]
    labels = [x[idx+num_batch] for idx in range(count)]
    num_items = len(data)
    train_data = torch.tensor(data, dtype=torch.float).reshape(num_items, num_batch, -1)
    train_labels = torch.tensor(labels, dtype=torch.float).reshape(num_items, -1)

    print(type(train_labels))
    return train_data, train_labels


def train(EPOCHS, net, X_train, y_train, criterion, optimizer, num_batch, hidden_size):
    losses = []

    for epoch in range(EPOCHS):
        print('epoch:', epoch)
        optimizer.zero_grad()
        hidden = torch.zeros(1, num_batch, hidden_size)
        output, hidden = net(X_train, hidden)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss.item() / len(X_train):.6f}')
        losses.append(loss.item() / len(X_train))

    return output, losses

def test(net, X_train, num_batch, hidden_size):

    net.eval()

    for i in range(400):
        hidden = torch.zeros(1, num_batch, hidden_size)
        output, hidden = net(X_train, hidden)

    return output

def main():

    num_div = 100  # 1周期の分割数
    cycles = 4  # 周期数
    num_batch = 25  # 1つの時系列データのデータ数
    X_train, y_train = make_train_data(num_div, cycles, num_batch)

    input_size = 1  # 入力サイズ
    hidden_size = 32  # 隠れ状態のサイズ
    output_size = 1  # 出力層のノード数
    net = Net(input_size, hidden_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    EPOCHS = 100

    _, losses = train(EPOCHS, net, X_train, y_train, criterion, optimizer, num_batch, hidden_size)
    output = test(net, X_train, num_batch, hidden_size)

    plt.plot(losses)
    plt.show()

    output = output.reshape(len(output)).detach()
    sample_data, _ = make_data(num_div, cycles)
    plt.plot(range(24, cycles * num_div), output)
    plt.plot(sample_data)
    plt.grid()

    plt.show()

if __name__ == '__main__':
    main()
