import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


class Encoder(torch.nn.Module):
    def __init__(self):
                                                                   #N, 3, 256, 256
        super().__init__()
        self.fc1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)  #N, 16, 128, 128
        self.fc2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)  #N, 32, 64, 64
        self.fc3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)  #N, 64, 32, 32
        self.fc4 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)  #N, 128, 16, 16
        self.fc5 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)  #N, 256, 8, 8
        self.fc6 = torch.nn.Conv2d(256, 512, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.ConvTranspose2d(512, 256, 8)  #N, 256, 8, 8
        self.fc2 = torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  #N, 128, 16, 16
        self.fc3 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  #N, 64, 32 32
        self.fc4 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  #N, 32, 64, 64
        self.fc5 = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)  #N, 16, 128, 128
        self.fc6 = torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1) #N, 3, 256, 256
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))  # -1～1に変換 sigmoid-> 0~1
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

#if using nn.maxpool2d, reverse is nn.MaxUnpool2d

class ImageFolder(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform
        img_list = []
        for path in self.img_paths:
            img = Image.open(path)
            img = self.transform(img)
            if img.shape[0]==1:
                continue
            img_list.append(img)
        self.img_list = img_list
        self.length = len(self.img_list)
    def __getitem__(self, index):


        return self.img_list[index]

    def _get_img_paths(self, img_dir):

        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in ImageFolder.IMG_EXTENSIONS
        ]

        return img_paths

    def __len__(self):

        return self.length


def get_dataset(batch_size):

    size = (256, 256)
    # Transform を作成する
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    # Dataset を作成する
    trainset = ImageFolder("train2014_small", transform)
    testset = ImageFolder("test2014_small", transform)
    # DataLoader を作成する
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=2)

    return trainloader, testloader

def imshow(img, title):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), )
    plt.title(title)
    plt.show()
#    plt.savefig('figure.png') # -----(2)

def train(net, criterion, optimizer, epochs, trainloader):
    losses = []
    output_and_label = []

    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}, ', end='')
        running_loss = 0.0
        for counter, img in enumerate(trainloader, 1):
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
    x = next(iterator)
    imshow(x, "sample")

    net = AutoEncoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    EPOCHS = 10

    output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)

    plt.plot(losses)

    output, org = output_and_label[-1]

    imshow(org, "original")
    imshow(output, "output")

if __name__ == '__main__':
    main()
