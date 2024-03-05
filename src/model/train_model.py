import argparse
import time

import torch.optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from src.features.dataset import FlickrDataset, content_transform, style_transform
from src.model.model import TransformNet
from src.model.vgg_loss import VGG16Loss
from src.utils.tools import normalize_for_vgg, gram_matrix_batch


def train(train_loader: DataLoader,
          transformer: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          vgg: nn.Module,
          mse_loss: nn.Module,
          gram_style: list,
          content_weight: int,
          style_weight: int,
          num_epochs: int,
          device: str):
    for epoch in range(num_epochs):
        transformer.train()
        vgg.eval()

        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, x in enumerate(tqdm(train_loader)):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)  # это наш контент
            y = transformer(x)  # это преобразованное изображение

            # теперь делаем нормализацию для vgg
            x = normalize_for_vgg(x)
            y = normalize_for_vgg(y)

            features_y = vgg(y)  # представления для преобразованного x
            features_x = vgg(x)  # представления для самого x

            # представление контента с одной фича мапы
            content_loss = content_weight * mse_loss(features_y[1], features_x[1])

            # для стиля со всех 4х
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix_batch(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])

            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % 40 == 0:
                mesg = "{} Epoch {}: content: {:.6f} style: {:.6f} total: {:.6f}".format(
                    time.ctime(), epoch + 1,
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1))
                print(mesg)


def fit(train_loader, style_img, content_weight: int, style_weight: int, num_epochs: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # основная сеть и оптимайзер
    transformer = TransformNet().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=int(len(train_dataset) / batch_size + 1) * num_epochs,
                                                           )

    # всё что у нас будет нужно для лосса
    vgg = VGG16Loss().to(device)
    mse_loss = nn.MSELoss()
    style_img = style_img.to(device)

    features_style = vgg(normalize_for_vgg(style_img))
    # Один раз считаем представления стиля с помощью vgg
    gram_style = [gram_matrix_batch(f) for f in features_style]

    train(train_loader=train_loader,
          transformer=transformer,
          optimizer=optimizer,
          scheduler=scheduler,
          vgg=vgg,
          mse_loss=mse_loss,
          gram_style=gram_style,
          content_weight=content_weight,
          style_weight=style_weight,
          num_epochs=num_epochs,
          device=device)

    torch.save(transformer.eval(), '../../models/model_mosaic.torch')
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.uint8)
    torch.onnx.export(
        transformer.eval(),
        dummy_input,
        '../../models/model_mosaic.onnx',
        input_names=['input'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'width', 3: 'height'}})


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_dir', default='../../data/Images', dest='data_dir')
    args_parser.add_argument('--style_path', default='../../data/mosaic.jpg', dest='style_path')
    args_parser.add_argument('--content_weight', default=1e5, type=int, dest='content_weight')
    args_parser.add_argument('--style_weight', default=2e10, type=int, dest='style_weight')
    args_parser.add_argument('--num_epochs', default=12, type=int, dest='num_epochs')
    args = args_parser.parse_args()

    batch_size = 16
    train_dataset = FlickrDataset(args.data_dir, content_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    style_img = Image.open(args.style_path).convert('RGB')
    style_img = style_transform(style_img)
    style_img = style_img.repeat(batch_size, 1, 1, 1)

    fit(
        train_loader,
        style_img,
        args.content_weight,
        args.style_weight,
        args.num_epochs,
    )
