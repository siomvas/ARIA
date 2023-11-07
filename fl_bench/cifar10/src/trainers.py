from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from .validator import Validator
from fl_bench.utils.AltScaffoldHelper import AltScaffoldHelper, get_lr_values
from lightly.utils.scheduler import CosineWarmupScheduler


class BaseTrainer(object):
    def __init__(self, train_config: Dict, method: str):
        self.init_lr = train_config["learning_rate"]
        self.max_steps = train_config["max_steps"]
        self.use_half_precision = train_config["use_half_precision"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)
        self.method = method
        self.optimizer_dict = train_config["optimizer_args"]
        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None
        self.fedprox_mu = train_config["fedprox_mu"]
        self.criterion_prox = None
        self.validator = Validator()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gradient_accumulation_steps = 1

    def configure_optimizer(self):
        optimizer_class = self.optimizer_dict.pop('class', None)
        assert optimizer_class in ['SGD', 'Adam', 'AdamW']
        self.opt = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.init_lr,
            **self.optimizer_dict)
        self.optimizer_dict['class'] = optimizer_class
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)

        self.sch = CosineWarmupScheduler(
            self.opt,
            warmup_epochs=200,
            max_epochs=self.max_steps/self.gradient_accumulation_steps,
            end_value=0.0,
        )
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def get_batch(self, data_loader: DataLoader, num_steps: int):
        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image, label = batch[0].to(device), batch[1].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    # self.scaler.unscale_(self.opt)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item()

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

    def setup(self, model, logger, abort_signal):
        self.model = model

        # global model setup for aux losses
        # self.global_model = deepcopy(model).eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def cleanup(self):
        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
    ):
        self.setup(model, logger, abort_signal)

        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()


class ScaffoldTrainer(BaseTrainer):
    def __init__(self, train_config: Dict, method: str):
        super().__init__(train_config, method)
        self.scaffold_helper = AltScaffoldHelper()

    def setup(self, model, logger, abort_signal):
        self.model = model
        # global model setup for aux losses
        self.global_model = deepcopy(model).eval()
        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        c_global_para, c_local_para = self.scaffold_helper.get_params()

        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")


            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image, label = batch[0].to(device), batch[1].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    # self.scaler.unscale_(self.opt)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    # SCAFFOLD model update step
                    curr_lr = get_lr_values(self.opt)[0]
                    self.scaffold_helper.model_update(
                        model=self.model,
                        curr_lr=curr_lr,
                        c_global_para=c_global_para,
                        c_local_para=c_local_para
                    )

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item() 

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # SCAFFOLD terms update step
            self.scaffold_helper.terms_update(
                model=self.model,
                curr_lr=curr_lr,
                c_global_para=c_global_para,
                c_local_para=c_local_para,
                model_global=self.global_model,
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

class MOONTrainer(BaseTrainer):
    def __init__(self, train_config: Dict, method: str):
        super().__init__(train_config, method)

    def setup(self, model, logger, abort_signal):
        self.model = model
        # global model setup for aux losses
        self.global_model = deepcopy(model).eval()
        self.logger = logger
        self.mu = 1
        self.tau = 0.5
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0

        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")


            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image, label = batch[0].to(device), batch[1].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    # self.scaler.unscale_(self.opt)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()


                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item() 

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )

            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

                # FROM old code
                # compute delta model, global model has the primary key set
                # local_weights = self.model.state_dict()

                # save to weight deque (FILO):
                # self.weights_local_old.appendleft(local_weights)
                        # make a copy of model_global as reference
                # model_global = copy.deepcopy(self.model)
                # for param in model_global.parameters():
                #     param.requires_grad = False
                
                # # initialize MOON old models as reference
                # models_local_old = []
                # for weight_local_old in self.weights_local_old:
                #     model_local_old = copy.deepcopy(self.model)
                #     for param in model_global.parameters():
                #         param.requires_grad = False
                #     model_local_old.load_state_dict(weight_local_old)
                #     models_local_old.append(model_local_old)
                # rep_local, outputs = self.model.rep_and_out(inputs)
                # loss = self.criterion(outputs, labels)

                # if current_round + 1 > self.warmup:
                #     # MOON loss term
                #     rep_global, _ = self.model_global.rep_and_out(inputs).detach()
                #     reps_local_old = []
                #     for old_model in models_local_old: 
                #         rep_local_old, _ = old_model.rep_and_out(inputs).detach()
                #         reps_local_old.append(rep_local_old)

                #     moon_loss = self.criterion_moon(rep_local, reps_local_old, rep_global)
                
                #     loss += moon_loss
                # self.criterion_moon = PTMOONLoss(mu=self.moon_mu, tau=self.moon_tau)
