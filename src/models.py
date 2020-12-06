import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=32):
      super(Generator, self).__init__()
      self.network = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),

          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),

          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),

          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )

    def forward(self, input):
      output = self.network(input)
      return output

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=32):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(

                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)

# class Generator(nn.Module):
#     def __init__(self, g_input_dim=100, g_output_dim=784):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(g_input_dim, 512)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
#         self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

#     # forward method
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         return torch.tanh(self.fc4(x))

# class Discriminator(nn.Module):
#     def __init__(self, d_input_dim=784):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(d_input_dim, 1024)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
#         self.fc4 = nn.Linear(self.fc3.out_features, 1)

#     # forward method
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.dropout(x, 0.3)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.dropout(x, 0.3)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         x = F.dropout(x, 0.3)
#         return torch.sigmoid(self.fc4(x))
