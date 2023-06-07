import os
import torch
import torch.optim as optim
from dotenv import load_dotenv
load_dotenv()
from fl_bench.networks.cifar10_nets import *
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from randomaug import RandAugment

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss
from pt.learners.cifar10_learner import CIFAR10Learner

CIFAR10_ROOT = os.getenv("CIFAR10_ROOT")

class CIFAR10LearnerViT(CIFAR10Learner): 
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        model_arch: str = "ViT",
        aggregation_epochs: int = 4,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        CIFAR10Learner.__init__(
            self,
            train_idx_root=train_idx_root,
            model_arch=model_arch,
            aggregation_epochs=aggregation_epochs,
            lr=lr,
            fedproxloss_mu=fedproxloss_mu,
            central=central,
            analytic_sender_id=analytic_sender_id,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def initialize(self, parts: dict, fl_ctx: FLContext):

        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )
        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        config = {
            "fl_args": fl_args,
            "client_id": self.client_id,
            "app_root": self.app_root,
            "fl_ctx": fl_ctx.get_all_public_props(),
        }

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(
            self.analytic_sender_id
        )  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = eval(self.model_arch + "()").to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=10e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.base_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        
        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)
        
        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        if aug:  
            N = 2; M = 14;
            self.transform_train.transforms.insert(0, RandAugment(N, M))

    def local_train(
        self,
        fl_ctx,
        train_loader,
        model_global,
        abort_signal: Signal,
        val_freq: int = 0,
    ):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )
            avg_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            self.writer.add_scalar(
                "train_loss", avg_loss / len(train_loader), current_step
            )
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(
                    self.valid_loader,
                    abort_signal,
                    tb_id="val_acc_local_model",
                    fl_ctx=fl_ctx,
                )
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)