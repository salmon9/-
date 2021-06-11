import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module):
    def __init__(self, input_size):
                                                                   #N, 1, 28, 28
        super().__init__()
        self.fc1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1)  #N, 16, 14, 14
        self.fc2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1) #N, 32, 7, 7
        self.fc3 = torch.nn.Conv2d(32, 64, 7) #N, 64, 1, 1
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = torch.nn.ConvTranspose2d(64, 32, 7)  #N, 32, 7, 7
        self.fc2 = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)  #N, 16, 14, 14
        self.fc3 = torch.nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)  #N, 1, 28, 28
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # -1～1に変換 sigmoid-> 0~1
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self, org_size):
        super().__init__()
        self.enc = Encoder(org_size)
        self.dec = Decoder(org_size)
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

#if using nn.maxpool2d, reverse is nn.MaxUnpool2d

def get_dataset(batch_size):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST('./data', train=True, transform=transform, download=True)
    testset = MNIST('./data', train=False, transform=transform, download=True)

    batch_size = batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#    plt.savefig('figure.png') # -----(2)

def train(net, criterion, optimizer, epochs, trainloader):
    losses = []
    output_and_label = []

    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}, ', end='')
        running_loss = 0.0
        for counter, (img, _) in enumerate(trainloader, 1):
            optimizer.zero_grad()
#            img = img.reshape(-1, input_size)
            output = net(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print('loss:', avg_loss)
        output_and_label.append((output, img))
    print('finished')
    return output_and_label, losses


def main():
    batch_size = 50

    trainloader, testloader = get_dataset(batch_size)

    iterator = iter(trainloader)
    x, _ = next(iterator)
    imshow(x)


    input_size = 28 * 28
    net = AutoEncoder(input_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    EPOCHS = 50

    output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)

    plt.plot(losses)

    output, org = output_and_label[-1]
    imshow(org)
    imshow(output)

if __name__ == '__main__':
    main()
