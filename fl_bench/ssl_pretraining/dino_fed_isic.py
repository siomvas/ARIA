import argparse
import copy
from typing import Optional, Tuple, Union, Dict, List
import os
import math

import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from fl_bench.networks.medmnist_nets import *
from lightly.data import LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.utils.scheduler import cosine_schedule, CosineWarmupScheduler
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms.dino_transform import DINOTransform


class DINO(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 512,
        model_name: str = 'resnet_18',
        seed: int = 1,
    ):
        super().__init__()
        self.model_name = model_name
        baseline = eval(model_name + f"({seed})")
        if self.model_name == "vit_b_16":
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
            input_dim = backbone.embed_dim
        elif self.model_name == "densenet_121":
            backbone = torch.nn.Sequential(*[
                baseline.features,
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            ])
        elif self.model_name == "nf_resnet50":
            backbone = baseline
            backbone.reset_classifier(0)
        else:
            backbone = torch.nn.Sequential(*list(baseline.children())[:-1])

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim=input_dim,
            output_dim=8192,
            freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim=input_dim,
            output_dim=8192)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=8192, warmup_teacher_temp=0.04, teacher_temp=0.04)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, max_epochs, 0.995, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        # self.logger.log_metrics({"Loss": loss}, step=self.current_epoch)
        self.output = teacher_out[0]

        return loss

    def on_train_start(self) -> None:
        self.avg_loss = 0.0
        self.avg_output_std = 0.0

    def on_train_epoch_end(self) -> None:
        output = self.output.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        # use moving averages to track the loss and standard deviation
        w = 0.9
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()
        collapse_level = max(0.0, 1 - math.sqrt(8192) * self.avg_output_std)
        self.log("Collapse Level", collapse_level, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.0005)
        scheduler = CosineWarmupScheduler(optim, warmup_epochs=10, end_value=2e-3, max_epochs=max_epochs)
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        wd = cosine_schedule(self.current_epoch, max_epochs, 4e-2, 1e-1)
        self.optimizers(use_pl_optimizer=False).param_groups[0]["weight_decay"] = wd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO training")
    parser.add_argument('--model_name', type=str, default='resnet_18', help='Name of the model to use')
    parser.add_argument('--input_dim', type=int, default=512, help='Input dimension of the model')
    parser.add_argument('--devices', type=str, default="auto", help='Devices to use for training')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder of unlabeled images')
    args = parser.parse_args()

    num_workers = 12
    batch_size = 128
    seed = 1
    max_epochs = 300
    path_to_data = args.data_dir

    pl.seed_everything(seed)
    model = DINO(model_name=args.model_name, input_dim=args.input_dim, seed=seed)
    transform = DINOTransform()
    dataset = LightlyDataset(path_to_data, transform=transform)
    # dataset.dataset.loader = lambda x: np.array(Image.open(x).convert("L"))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=args.devices,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        log_every_n_steps=1,
        default_root_dir=f"{os.getcwd()}/lightning_logs/{args.model_name}",
        precision="bf16-mixed",
        use_distributed_sampler=True,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)
