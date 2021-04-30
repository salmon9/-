import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class Model (nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.l1 = nn.Linear(2,20)
    self.l2 = nn.Linear(20,40)
    self.l3 = nn.Linear(40,2)

  def forward(self,x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.l3(x)
    return x

class my_dataset(torch.utils.data.Dataset):
  def __init__(self, point):

    self.point = point
    self.len = len(point)  

  def __getitem__(self, index):

    input = self.point[index-1] + np.random.normal(scale=0.001)  # indexの扱いに気を付ける
    target = self.point[index]  + np.random.normal(scale=0.001)  # 最初と同じ点にならないように

    return input, target

  def __len__(self):
    return self.len

def get_dataset(n):

    parameter = np.radians(np.linspace(0,360,n))  # 0〜360のランダム値 
    x = np.cos(parameter)
    y = np.sin(parameter)

    dataset = np.array([x,y]).T

    return dataset, x, y

def train(data_loader, model, criterion, optimizer, epoch):
    
    train_loss_list = []
    for t in range(epoch):

        train_loss = 0
        for i, (input, target) in enumerate(data_loader):  # 1ミニバッチずつ計算

            model.zero_grad()

            output = model(input.float())

            loss = criterion(output, target.float())
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            print(t, loss.item())

        train_loss_list.append(train_loss)

    drawing_loss_graph(epoch, train_loss_list)
    
def test(first_point, model):

    record_output = np.array([[1.0, 0.0]])

    for i in range(150):

        test_output = model(first_point)
        first_point = test_output
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

    data_number = 100  #　準備するデータの数
    batch_size  = 20  # 1つのミニバッチのデータの数

    points, x, y = get_dataset(data_number)
    dataset = my_dataset(points)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = Model()
    criterion = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 500

    first_point = [[1.0,0.0]]
    first_point = torch.Tensor(first_point)
        
    model.train()
    train(data_loader, model, criterion, optimizer, epoch)

    model.eval()
    final_test_output = test(first_point, model)

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
