import copy
import os
from dotenv import load_dotenv
import numpy as np
import torch
import torch.optim as optim
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import (AppConstants, ModelName,
                                             ValidateType)
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from fl_bench.networks.cifar10_nets import *
from fl_bench.utils.cifar10_helpers import CIFAR10_Idx
from fl_bench.utils.PTMOONLoss import PTMOONLoss
from collections import deque


load_dotenv()
CIFAR10_ROOT = os.getenv("CIFAR10_ROOT")

# HACK: Instead of custom sng workflow, we keep the current z as z_prev as client attribute
# this breaks if num_threads < num_clients
class CIFAR10MoonLearner(Learner):
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        model_arch: str = "ModerateCNN",
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        moon_mu: float = 5.0,
        moon_tau: float = 0.5,
        warmup: int = 1,
        buffer_len: int = 1,
        central: bool = False,
        save_freq: int = 10,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):

        super().__init__()
        self.train_idx_root = train_idx_root
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.model_arch = model_arch
        self.fedproxloss_mu = fedproxloss_mu

        self.moon_mu = moon_mu
        self.moon_tau = moon_tau
        self.warmup = warmup
        self.buffer_len = buffer_len

        self.best_acc = 0.0
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.writer = None
        self.analytic_sender_id = analytic_sender_id
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.weights_local_old = None
        self.optimizer = None
        self.criterion = None
        self.criterion_prox = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

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

        self.writer = parts.get(self.analytic_sender_id)
        if not self.writer:
            self.writer = SummaryWriter(self.app_root)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if "moon" not in self.model_arch:
            self.log_info(fl_ctx, f"Only use _moon models")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.model = eval(self.model_arch + "()").to(self.device)
        self.weights_local_old = deque(maxlen=self.buffer_len)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)

        self.log_info(fl_ctx, f"Using MOON loss with mu {self.moon_mu} and tau {self.moon_tau}")
        self.criterion_moon = PTMOONLoss(mu=self.moon_mu, tau=self.moon_tau)

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

    def _create_datasets(self, fl_ctx: FLContext):

        if self.train_dataset is None or self.train_loader is None:
            if not self.central:
                # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
                site_idx_file_name = os.path.join(
                    self.train_idx_root, self.client_id + ".npy"
                )
                self.log_info(fl_ctx, f"IndexList Path: {site_idx_file_name}")
                if os.path.exists(site_idx_file_name):
                    self.log_info(fl_ctx, "Loading subset index")
                    site_idx = np.load(
                        site_idx_file_name
                    ).tolist()  # TODO: get from fl_ctx/shareable?
                else:
                    self.system_panic(
                        f"No subset index found! File {site_idx_file_name} does not exist!",
                        fl_ctx,
                    )
                    return
                self.log_info(fl_ctx, f"Client subset size: {len(site_idx)}")
            else:
                site_idx = None  # use whole training dataset if self.central=True

            self.train_dataset = CIFAR10_Idx(
                root=CIFAR10_ROOT,
                data_idx=site_idx,
                train=True,
                download=False,
                transform=self.transform_train,
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        if self.valid_dataset is None or self.valid_loader is None:
            self.valid_dataset = datasets.CIFAR10(
                root=CIFAR10_ROOT,
                train=False,
                download=False,
                transform=self.transform_valid,
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def finalize(self, fl_ctx: FLContext):
        self.writer.flush()

    def local_train(
        self,
        fl_ctx,
        train_loader,
        model_global,
        models_local_old,
        current_round,
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
                self.optimizer.zero_grad()

                rep_local, outputs = self.model.rep_and_out(inputs)
                loss = self.criterion(outputs, labels)

                if current_round + 1 > self.warmup:
                    # MOON loss term
                    rep_global, _ = self.model_global.rep_and_out(inputs).detach()
                    reps_local_old = []
                    for old_model in models_local_old: 
                        rep_local_old, _ = old_model.rep_and_out(inputs).detach()
                        reps_local_old.append(rep_local_old)

                    moon_loss = self.criterion_moon(rep_local, reps_local_old, rep_global)
                
                    loss += moon_loss

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
                    tb_id="val_metric_local_model",
                    fl_ctx=fl_ctx,
                )
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def train(self, shareable, fl_ctx, abort_signal) -> Shareable:
        self._create_datasets(fl_ctx)
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(
            fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}"
        )
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(
                        weights, local_var_dict[var_name].shape
                    )
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False
        
        # initialize MOON old models as reference
        models_local_old = []
        for weight_local_old in self.weights_local_old:
            model_local_old = copy.deepcopy(self.model)
            for param in model_global.parameters():
                param.requires_grad = False
            model_local_old.load_state_dict(weight_local_old)
            models_local_old.append(model_local_old)

        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            models_local_old=models_local_old,
            abort_signal=abort_signal,
            current_round=current_round,
            val_freq=1 if self.central else 0,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc = self.local_valid(
            self.valid_loader, abort_signal, tb_id="val_metric_local_model", fl_ctx=fl_ctx
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_metric_local_model: {acc:.4f}")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()

        # save to weight deque (FILO):
        self.weights_local_old.appendleft(local_weights)
        # prepare delta to send to server
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except BaseException as e:
                raise ValueError("Unable to load best model") from e

            # Create DXO and shareable from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_weights)
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(
                    fl_ctx,
                    f"best local model not found at {self.best_local_model_file}.",
                )
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(
                f"Unknown model_type: {model_name}"
            )  # Raised errors are caught in LearnerExecutor class.

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _i, (inputs, labels) in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred_label = torch.max(outputs.data, 1)

                total += inputs.data.size()[0]
                correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.epoch_global)

        return metric

    def validate(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        self._create_datasets(fl_ctx)

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(
            AppConstants.MODEL_OWNER
        )
        if model_owner:
            self.log_info(
                fl_ctx,
                f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}",
            )
        else:
            model_owner = "global_model"  # evaluating global model during training

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(
                        torch.reshape(weights, local_var_dict[var_name].shape)
                    )
                    n_loaded += 1
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(
                f"No weights loaded for validation! Received weight dict is {global_weights}"
            )

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_acc = self.local_valid(
                self.valid_loader,
                abort_signal,
                tb_id="val_metric_global_model",
                fl_ctx=fl_ctx,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_global_model ({model_owner}): {global_acc}")

            return DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_acc},
                meta={},
            ).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # perform valid
            train_acc = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training acc ({model_owner}): {train_acc}")

            val_metric = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc ({model_owner}): {val_metric}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_metric}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
