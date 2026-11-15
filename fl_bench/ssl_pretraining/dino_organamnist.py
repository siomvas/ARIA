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

class GaussianNoise:
    """Applies random Gaussian noise to a tensor.

    The intensity of the noise is dependent on the mean of the pixel values.
    See https://arxiv.org/pdf/2101.04909.pdf for more information.

    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)
        sigma = mu / snr
        noise = torch.normal(torch.zeros(sample.shape), sigma)
        return sample + noise


class my_DINOViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = 10,
        gaussian_blur: float = 1.0,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        transform = [
            T.Grayscale(3),
            T.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                interpolation=PIL.Image.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            GaussianBlur(
                kernel_size=kernel_size,
                scale=kernel_scale,
                sigmas=sigmas,
                prob=gaussian_blur,
            ),
            T.ToTensor(),
            # GaussianNoise(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image) -> torch.Tensor:
        transformed: torch.Tensor = self.transform(image)
        return transformed


class my_DINOTransform(MultiViewTransform):
    """Implements the global and local view augmentations for DINO [0].

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        kernel_scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
        gaussian_blur: Tuple[float, float, float] = (1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2)
    ):
        # first global crop
        global_transform_0 = my_DINOViewTransform(
            gaussian_blur=gaussian_blur[0],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas
        )

        # second global crop
        global_transform_1 = my_DINOViewTransform(
            gaussian_blur=gaussian_blur[1],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas
        )

        # transformation for the local small crops
        local_transform = my_DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            gaussian_blur=gaussian_blur[2],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
        )
        local_transforms = [local_transform] * n_local_views
        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)


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

        self.criterion = DINOLoss(output_dim=8192, warmup_teacher_temp=0.02, teacher_temp=0.04)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 300, 0.9995, 1)
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
        scheduler = CosineWarmupScheduler(optim, warmup_epochs=10, end_value=2e-3, max_epochs=300)
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        wd = cosine_schedule(self.current_epoch, 300, 4e-2, 1e-1)
        self.optimizers(use_pl_optimizer=False).param_groups[0]["weight_decay"] = wd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO training")
    parser.add_argument('--model_name', type=str, default='resnet_18', help='Name of the model to use')
    parser.add_argument('--input_dim', type=int, default=512, help='Input dimension of the model')
    parser.add_argument('--devices', type=str, default="auto", help='Devices to use for training')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder of unlabeled images')
    args = parser.parse_args()

    num_workers = 8
    batch_size = 128
    seed = 1
    max_epochs = 100
    path_to_data = args.data_dir

    pl.seed_everything(seed)
    model = DINO(model_name=args.model_name, input_dim=args.input_dim, seed=seed)
    transform = my_DINOTransform()
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


# fix every scheduling to 300 epochs but run for fewer.