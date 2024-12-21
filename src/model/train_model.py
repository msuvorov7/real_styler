import argparse
import multiprocessing

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

import pytorch_lightning as pl

from src.features.dataset import FlickrDataset, content_transform, style_transform
from src.model.model import TransformNetLightning, TransformNet, StyleLoss


multiprocessing.set_start_method('fork')


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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

    style_img = Image.open(args.style_path).convert('RGB')
    style_img = style_transform(style_img)
    style_img = style_img.repeat(batch_size, 1, 1, 1)

    trainer = pl.Trainer(
        accelerator='mps',
        # logger=logger,
        # callbacks=[checkpoint_callback, ],
        max_epochs=args.num_epochs,
        accumulate_grad_batches=1,
        precision="16-mixed",
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=5,
        use_distributed_sampler=True,
    )

    model = TransformNetLightning(
        model=TransformNet(),
        criterion=StyleLoss(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            style_img=style_img,
            device='mps',
        ),
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    # save model
    torch.save(model.model.eval(), 'models/model_mosaic.torch')
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.uint8)
    torch.onnx.export(
        model.model.to('cpu').eval(),
        dummy_input,
        'models/model_mosaic_2.onnx',
        input_names=['input'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'width', 3: 'height'}})
