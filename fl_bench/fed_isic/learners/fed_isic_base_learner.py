import json
import os
import torch
from dotenv import load_dotenv
from monai.inferers.inferer import SimpleInferer
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from sklearn.metrics import balanced_accuracy_score
import torch.optim

from fl_bench.fed_isic.learners.my_learner import MyLearner
from fl_bench.networks.fed_isic_nets import *
from fl_bench.utils.FedIsicDM import FedIsicDataModule
from fl_bench.utils.PTMOONLoss import PTMOONLoss
from fl_bench.utils.WeightedFocalLoss import WeightedFocalLoss

load_dotenv()


class FedIsicLearner(MyLearner):
    def __init__(
        self,
        train_config_filename,
        train_task_name: str = AppConstants.TASK_TRAIN,
        central: bool = False
    ):

        super().__init__(train_task_name=train_task_name)
        self.train_config_filename = train_config_filename
        self.central = central
        self.config_info = None

    def train_config(self, fl_ctx):

        self.config_info = self.load_config_json(fl_ctx)
        self.lr = self.config_info["learning_rate"]
        self.epochs = self.config_info["epochs"]
        self.model_name = self.config_info["model_name"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            raise ValueError("No GPU detected")
        self.model = eval(self.model_name + "()").to(self.device)
        self.central = self.config_info["central"]
        optimizer_dict = self.config_info["optimizer_args"]
        optimizer_class = optimizer_dict.pop('class', None)
        assert optimizer_class in ['SGD', 'Adam', 'AdamW']
        self.optimizer = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.lr,
            **optimizer_dict)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
        )
        self.fedprox_mu = self.config_info["fedprox_mu"]
        self.moon_mu = self.config_info["moon_mu"]

        self.criterion = WeightedFocalLoss()
        if self.fedprox_mu > 0:
            self.log_info(fl_ctx, f"use FedProx w/ mu {self.fedprox_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedprox_mu)
        if self.moon_mu > 0:
            self.log_info(fl_ctx, f"MOON w/ mu = {self.moon_mu}")
            self.criterion_mu = PTMOONLoss(mu=self.moon_mu)

        # Set DataModule
        dataset_base_dir = os.getenv("ISIC_ROOT")
        self.data_module = FedIsicDataModule(
            dataset_base_dir,
            client_idx=fl_ctx.get_identity_name(),
            central=self.central
            )

        self.t_len, self.v_len = (
            len(self.data_module.train_dataloader),
            len(self.data_module.val_dataloader)
            )
        self.log_info(fl_ctx, f"T Size: {self.t_len}, V Size: {self.v_len}")

        # Set inferer and evaluation metric
        self.inferer = SimpleInferer()
        self.valid_metric = balanced_accuracy_score

    def load_config_json(self, fl_ctx):
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(
            app_config_dir, self.train_config_filename
        )
        if not os.path.isfile(train_config_file_path):
            self.log_error(fl_ctx, f"No config at {train_config_file_path}")
        with open(train_config_file_path) as file:
            config_info = json.load(file)
        return config_info

    def validate(self, shareable, fl_ctx, abort_signal):

        # Handle boilerplate + loading before local_valid
        self.fl_callback.on_validation_start(self, shareable, abort_signal)
        y = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for _, batch_data in enumerate(self.data_module.val_dataloader):
                self.fl_callback.on_validation_batch_start(abort_signal)

                val_images = batch_data["image"].to(self.device)
                val_labels = batch_data["label"]
                y.append(val_labels)
                # Inference
                outputs = self.inferer(val_images, self.model)
                _, outputs_categorical = torch.max(outputs, 1)
                y_pred.append(outputs_categorical)

            y = torch.cat(y)
            y_pred = torch.cat(y_pred)
            metric_score = self.valid_metric(y.cpu(), y_pred.cpu())

        self.fl_callback.on_validation_end(
            self, abort_signal, fl_ctx, metric_score)