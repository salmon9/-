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
#        x = torch.tanh(self.fc6(x))  # -1～1に変換 sigmoid-> 0~1
        x = self.fc6(x)
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
    IMG_EXTENSIONS = [".jpg"]  #拡張子指定

    def __init__(self, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(img_dir)  #画像のディレクトリを入力
        self.transform = transform  #transformの定義
        img_list = []  #返す画像のデータセットの宣言
        for path in self.img_paths:
            img = Image.open(path)  #画像読み込み
            img = self.transform(img)  #画像処理
            if img.shape[0]==1:  #(channel, height, width)のうちchannel=1(monocolour)を弾く
                continue  #channel==1の時この後の行は実行されない＝appendされない
            img_list.append(img)  #弾いた写真以外をappend
        self.img_list = img_list
        self.length = len(self.img_list)  #長さは返すデータセットのもの、元本のではない
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

    size = 256
    # Transform を作成する
    #Composeは［ ］の中を順番に 実行
    transform = transforms.Compose([transforms.RandomResizedCrop(size), transforms.ToTensor()])
    #RandomResizedCropは(size,size)の大きさにcropとresizeしてくれる
    #size=(height,width)も可　
    #ToTensorは(height, width, channel)を(channel, height, width)に変えてくれる
    #元々のデータはndarray

    # Dataset を作成する
    trainset = ImageFolder("train2014_small", transform)
    testset = ImageFolder("test2014_small", transform)
    # DataLoader を作成する
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=2)
    #batch_sizeは一回の学習になんこのデータを使うか
    #num_workersは複数処理をするかどうか
    return trainloader, testloader

def imshow(img, title):
    img = torchvision.utils.make_grid(img)
    #複数の画像を一枚に合成する関数
    #入力tensor,出力tensor
    print(img.shape)
    img = img / 2 + 0.5 #色彩調整？？
    print(img.shape)
    npimg = img.detach().numpy()
    #tensorからndarrayに変換
    #ndarrayじゃないとimshowで出力できない
    plt.imshow(np.transpose(npimg, (1, 2, 0)), )
    #(channel, height, width)→(height, width, channel)
    plt.title(title)
    plt.show()
#    plt.savefig('figure.png') # -----(2)

def train(net, criterion, optimizer, epochs, trainloader):
    losses = []  #lossを入れる配列
    output_and_input = []  #ouputとinputを一緒に格納する配列

    for epoch in range(1, epochs+1):  #epoch回す
        print(f'epoch: {epoch}, ', end='')
        running_loss = 0.0  #epoch毎にlossを0にリセット
        for counter, input_img in enumerate(trainloader):  #インデックス、要素を取り出す
            optimizer.zero_grad()  #最適化対象のすべてのパラメータの勾配を0にする
#            input_img = input_img.reshape(-1, input_size)  #cnnだといらない
            output = net(input_img)  #学習モデルにimgを入力、出力結果がoutput
            loss = criterion(output, input_img)  #outputとinputを比較してlossを取得
            loss.backward()  #損失関数の逆伝播
            optimizer.step()  #重みの更新
            running_loss += loss.item  #epoch毎のlossを計算したいからbatch毎に加算
        avg_loss = running_loss / counter  #毎epochの平均loss
        losses.append(avg_loss)  #matplotlibでlossの変化のグラフを出すために使用
        print('loss:', avg_loss)  #lossの表示
        output_and_input.append((output, input_img))  #outputとinputを一緒に格納
    print('finished')
    return output_and_input, losses

def main():
    batch_size = 50

    trainloader, testloader = get_dataset(batch_size)  #train用とtest用のdatasetを作成

    iterator = iter(trainloader)
    x = next(iterator)
    imshow(x, "sample")  #一旦読み込んだdatasetの例を表示

    net = AutoEncoder()  #学習モデルの宣言
    criterion = torch.nn.MSELoss()  #損失関数の定義
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  #最適化手法の定義
    EPOCHS = 20

    output_and_input, losses = train(net, criterion, optimizer, EPOCHS, trainloader)

    plt.plot(losses)

    output, input_img = output_and_input[-1]

    imshow(input_img, "original")
    imshow(output, "output")

if __name__ == '__main__':
    main()
