import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchsummary import summary
from omegaconf import DictConfig


from pathlib import Path
from logging import getLogger
import dataclasses
from dataclasses import dataclass


class Solver:

    def __init__(self, generator, discriminator, dataloader,
            g_optim, d_optim, epochs, device, save_dir):

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir)
        Path("img").mkdir(parents=True, exist_ok=True)
        self.img_save = Path("img")

        torch.backends.cudnn.benchmark = True

        self.writer = SummaryWriter(log_dir=".")
        self.logger = getLogger("Solver")

        summary(self.generator, (100, 1, 1))
        summary(self.discriminator, (1,28,28))

    def save(self):
        torch.save(self.generator.state_dict(), str(self.save_dir / " generator.pt"))
        torch.save(self.discriminator.state_dict(), str(self.save_dir / "discriminator.pt"))

    def save_im(self, img, x, name, epoch):
        # img = torch.reshape(img, (img.size()[0],1,28,28))
        Path(self.img_save / "epoch_{}".format(epoch)).mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image((img[0] + 1) / 2, str(self.img_save / "epoch_{}".format(epoch) / (name + ".png")))
        np.save(str(self.img_save / "epoch_{}".format(epoch) / (name + ".npz")), x[0].cpu().numpy())

    def test (self):
        pass

    def train(self):
        for epoch in range(self.epochs):
            res = self._train_epoch(epoch)
            for k,v in res.items():
                self.logger.info("epoch{}\t{}:{:.4f}".format(epoch, k,v))
                self.writer.add_scalar(k,v,epoch)
            # self.scheduler.step()
        return res

    def _gen_noise(self, input):
        return torch.randn(input.size()[0], 100, 1, 1).to(self.device)

    def _train_epoch(self, epoch):
        d_real_loss = 0
        d_fake_loss = 0
        num_data = 0
        loss = nn.BCELoss()
        for idx, (image, _) in enumerate(self.dataloader):
            image = image.to(self.device)
            # image = torch.flatten(image, start_dim=1)

            # train D
            self.d_optim.zero_grad()
            input = self._gen_noise(image)

            # fake
            fake = self.generator(input)
            d_fake = self.discriminator(fake)
            # d_fake = self.discriminator(fake).view(-1)

            # real
            d_real = self.discriminator(image)
            # d_real = self.discriminator(image).view(-1)

            # calculate loss
            real_label = torch.full(d_real.size(), 1., dtype=torch.float, device=self.device)
            fake_label = torch.full(d_fake.size(), 0., dtype=torch.float, device=self.device)

            fake_loss = nn.BCELoss()(d_fake, fake_label)
            real_loss = nn.BCELoss()(d_real, real_label)
            loss = fake_loss + real_loss

            # update
            loss.backward()
            self.d_optim.step()

            # train G
            self.g_optim.zero_grad()

            fake = self.generator(input)
            d_fake = self.discriminator(fake)
            g_fake_loss = nn.BCELoss()(d_fake, real_label)
            g_fake_loss.backward()
            self.g_optim.step()

            num_data += 1

            d_real_loss += real_loss
            d_fake_loss += fake_loss + g_fake_loss

            if idx %100 == 0:
                self.save_im(fake, input, str(epoch) + str(idx), epoch)

        d_real_loss /= num_data
        d_fake_loss /= 2 * num_data
        return {"D(img)": d_real_loss.sum().item(), "D(G(x))":d_fake_loss.sum().item()}
