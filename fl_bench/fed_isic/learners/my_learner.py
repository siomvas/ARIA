from abc import abstractmethod

import numpy as np
import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants

from fl_bench.utils.FedCallback import FedCallback


class MyLearner(Learner):
    def __init__(
            self,
            train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.train_task_name = train_task_name
        self.best_metric = 0.0
        self.epoch_of_start_time = 0
        self.epoch_global = 0
        self.fl_callback = FedCallback()
        self.global_weights = None
        self.model_global = None
        self.epochs = None

    @abstractmethod
    def train_config(self, fl_ctx):
        raise NotImplementedError

    def initialize(self, parts, fl_ctx):

        # handle boilerplate
        self.fl_callback.setup(self, fl_ctx)
        self.train_config(fl_ctx)

    def train(
        self,
        shareable,
        fl_ctx,
        abort_signal
    ) -> Shareable:

        # Handle all boilerplate before local_train
        self.fl_callback.on_train_start(
            shareable,
            self,
            fl_ctx,
            abort_signal)

        for epoch in range(self.epochs):
            self.fl_callback.on_train_epoch_start(
                self,
                fl_ctx,
                epoch,
                abort_signal
            )
            for i, batch_data in enumerate(self.data_module.train_dataloader):
                self.fl_callback.on_train_batch_start(abort_signal)
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if self.fedprox_mu > 0:
                    prox_loss = self.criterion_prox(self.model, self.model_global)
                    loss += prox_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_step = self.t_len * self.epoch_global + i
                self.writer.add_scalar("train_loss", loss.item(), current_step)

        self.epoch_of_start_time += self.epochs

        # What to send to server
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in self.global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(
                local_weights[name].cpu().numpy(),
                self.global_weights[name],
                dtype=np.float32,
            )
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.writer.flush()
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, self.t_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable, fl_ctx, abort_signal):

        # # Handle boilerplate + loading before local_valid
        # self.fl_callback.on_validation_start(self, shareable, abort_signal)

        # self.model.eval()
        # with torch.no_grad():
        #     metric = 0
        #     for i, batch_data in enumerate(self.data_module.val_dataloader):
        #         self.fl_callback.on_validation_batch_start(abort_signal)

        #         val_images = batch_data["image"].to(self.device)
        #         val_labels = batch_data["label"].to(self.device)
        #         # Inference
        #         outputs = self.inferer(val_images, self.model)
        #         _, outputs_categorical = torch.max(outputs, 1)
        #         metric_score = self.valid_metric(val_labels.cpu(), outputs_categorical.cpu())
        #         metric += metric_score.item()
        #     # compute mean metric over whole validation set
        #     metric /= self.v_len

        # self.fl_callback.on_validation_end(
        #     self, abort_signal, fl_ctx, metric)
        pass

    def finalize(self):
        self.writer.flush()
