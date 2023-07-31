import copy

import numpy as np
import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from torch.utils.tensorboard.writer import SummaryWriter


class SurgicalCallback():
    """Callback to handle federated logic during a training pass

    """
    def setup(self, learner, fl_ctx) -> None:
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        learner.client_id = fl_ctx.get_identity_name()
        learner.log_info(
            fl_ctx,
            f"Client {learner.client_id} initialized with args: \n {fl_args}",
        )
        learner.writer = SummaryWriter(app_dir)

    def on_train_batch_start(self, abort_signal):
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

    def on_train_epoch_start(
            self,
            learner,
            fl_ctx,
            epoch,
            abort_signal
    ):
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        learner.log_info(
            fl_ctx,
            f"{learner.client_id}:{epoch+1}/{learner.epochs}(lr={learner.lr})")
        learner.epoch_global = learner.epoch_of_start_time + epoch

    def on_validation_batch_start(
        self,
        abort_signal
    ):
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

    def on_train_start(
            self,
            shareable,
            learner,
            fl_ctx,
            abort_signal
    ):
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        learner.log_info(
            fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}"
        )
        learner.log_info(fl_ctx, f"Client id: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        if learner.epoch_global == 0:
            learner.global_weights = dxo.data
        else:
            learner.global_weights = dxo.data[0]

        local_var_dict = learner.model.state_dict()
        model_keys = learner.global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = learner.global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    learner.global_weights[var_name] = np.reshape(
                        weights, local_var_dict[var_name].shape
                    )
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(
                        learner.global_weights[var_name]
                    )
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )
        learner.model.load_state_dict(local_var_dict)

        # local steps
        learner.log_info(fl_ctx, f"Local steps per epoch: {learner.t_len}")

        # make a copy of model_global as reference for potential FedProx loss
        if learner.fedprox_mu > 0 or learner.moon_mu > 0:
            learner.model_global = copy.deepcopy(learner.model)
            for param in learner.model_global.parameters():
                param.requires_grad = False
        else:
            learner.model_global = None

    def on_validation_start(
            self,
            learner,
            shareable,
            abort_signal
    ):
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        # update local model weights with received weights
        dxo = from_shareable(shareable)
        if learner.epoch_global == 0:
            global_weights = dxo.data
        else:
            global_weights = dxo.data[0]
        local_var_dict = learner.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(
                    global_weights[var_name],
                    device=learner.device
                    )
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(
                        torch.reshape(weights, local_var_dict[var_name].shape)
                    )
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )
        learner.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError("No weights loaded for validation!")

    def on_validation_end(
            self,
            learner,
            abort_signal,
            fl_ctx,
            global_metric
    ) -> Shareable:

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        learner.log_info(
            fl_ctx, f"val_metric_global_model: {global_metric:.4f}")
        learner.writer.add_scalar(
            "val_metric_global_model",
            global_metric,
            learner.epoch_global
        )

        metric_dxo = DXO(
            data_kind=DataKind.METRICS,
            data={MetaKey.INITIAL_METRICS: global_metric},
            meta={},
        )
        metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, learner.v_len)
        return metric_dxo.to_shareable()
