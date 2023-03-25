import torch.nn as nn
from torchsummary import summary




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.activ = nn.SELU()

        self.conv1 = nn.ConvTranspose2d(in_channels=50, out_channels=256, kernel_size=(3,2), stride=2, padding=0)

        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,2), stride=2, padding=0)

        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3,2), stride=2, padding=0)

        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3,2), stride=2, padding=0, output_padding=(1,1))

        self.conv5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3,2), stride=(2,1), padding=1, output_padding=(1,0))
        



    def forward(self, input):
        x = self.activ(self.conv1(input))
        x = self.activ(self.conv2(x))
        x = self.activ(self.conv3(x))
        x = self.activ(self.conv4(x))
        x = self.activ(self.conv5(x))
        return x







class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.activ = nn.SELU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,2), stride=(2,1), padding=1)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3,2), stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3,2), stride=2, padding=0)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,2), stride=2, padding=0)

        self.fc = nn.Linear(in_features=1, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.activ(self.conv1(input))
        x = self.activ(self.conv2(x))
        x = self.activ(self.conv3(x))
        x = self.activ(self.conv4(x))
        x = x.view(-1, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x







if __name__ == "__main__":
    gen = Generator()
    summary(gen, (50, 1, 1))

    disc = Discriminator()
    summary(disc, (1, 64, 16))