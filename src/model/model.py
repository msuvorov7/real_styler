import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.model.vgg_loss import VGG16Loss
from src.utils.tools import normalize_for_vgg, gram_matrix_batch


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 ):
        super(ConvBlock, self).__init__()
        to_pad = kernel_size // 2
        # чтобы не было артефактов на краях используем ReflectionPad2d
        self.reflection_pad = nn.ReflectionPad2d(to_pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 scale: int
                 ):
        super(UpsampleBlock, self).__init__()
        to_pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(to_pad)
        # чтобы не было шахматных артефактов, используем на выходе свертку
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.reflection_pad(self.upsample(x)))


class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )

        self.residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.decoder = nn.Sequential(
            UpsampleBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, scale=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            UpsampleBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, scale=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=32, out_channels=3, kernel_size=9, stride=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        residual = self.residual(encoded)
        decoded = self.decoder(residual)
        return decoded



class TransformNetLightning(pl.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, batch):
        output = self.model(batch)
        
        # eval loss
        loss = self.criterion(
            x=batch,
            y=output,
        )

        return loss
    
    def training_step(self, batch, batch_idx):
        train_loss = self.evaluate(batch)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=int(len(train_dataset) / batch_size + 1) * num_epochs,
        # )
        return optimizer
    

class StyleLoss(nn.Module):
    def __init__(
            self,
            content_weight,
            style_weight,
            style_img,
            device,
        ):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        self.loss = None
        self.vgg = VGG16Loss().to(device)
        self.mse_criterion = nn.MSELoss()

        features_style = self.vgg(normalize_for_vgg(style_img.to(device)))
        self.gram_style = [gram_matrix_batch(f) for f in features_style]

    def forward(self, x, y):
        n_batch = len(x)

        # теперь делаем нормализацию для vgg
        x = normalize_for_vgg(x)
        y = normalize_for_vgg(y)

        features_y = self.vgg(y)  # представления для преобразованного x
        features_x = self.vgg(x)  # представления для самого x
    
        # представление контента с одной фича мапы
        content_loss = self.content_weight * self.mse_criterion(features_y[1], features_x[1])

        # для стиля со всех 4х
        style_loss = 0.0
        for ft_y, gm_s in zip(features_y, self.gram_style):
            gm_y = gram_matrix_batch(ft_y)
            style_loss += self.mse_criterion(gm_y, gm_s[:n_batch, :, :])

        style_loss *= self.style_weight

        self.loss = content_loss + style_loss
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss