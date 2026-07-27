"""nnU-Net trainer with MMSeg-like validation tracking.

The standard nnU-Net loss is retained: Dice + cross entropy for label maps
(Dice + BCE for region training). Early stopping and the best checkpoint use
the non-smoothed foreground mean Dice.
"""

import os
from os.path import join
from time import time
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainerDiceEarlyStoppingTensorboard(nnUNetTrainer):
    """Track classwise Dice/IoU and stop on validation ``fg_mDice``.

    Configuration is intentionally done through environment variables because
    nnU-Net trainer classes have a fixed constructor:

    ``NNUNET_EARLY_STOPPING_PATIENCE`` (default 15),
    ``NNUNET_EARLY_STOPPING_MIN_DELTA`` (default 0.0001).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping_patience = int(
            os.getenv("NNUNET_EARLY_STOPPING_PATIENCE", "15")
        )
        self.early_stopping_min_delta = float(
            os.getenv("NNUNET_EARLY_STOPPING_MIN_DELTA", "0.0001")
        )
        if self.early_stopping_patience < 1:
            raise ValueError("NNUNET_EARLY_STOPPING_PATIENCE must be >= 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("NNUNET_EARLY_STOPPING_MIN_DELTA must be >= 0")

        self._best_fg_mdice = -np.inf
        self._epochs_without_improvement = 0
        self._early_stop_requested = False
        self._tb_writer = None
        self._best_metrics = {}

    def _class_names(self, count: int) -> List[str]:
        if self.label_manager.has_regions:
            candidates = [
                name
                for name, value in self.dataset_json["labels"].items()
                if name.lower() != "background"
                and isinstance(value, (list, tuple))
            ]
        else:
            candidates = [
                name
                for name, value in sorted(
                    self.dataset_json["labels"].items(),
                    key=lambda item: int(item[1]),
                )
                if int(value) != 0
            ]
        return [
            str(candidates[i] if i < len(candidates) else f"class_{i + 1}")
            .replace("/", "_")
            for i in range(count)
        ]

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard is required by this trainer. Install it with "
                    "`pip install tensorboard`."
                ) from exc
            self._tb_writer = SummaryWriter(join(self.output_folder, "tensorboard"))
            self._tb_writer.add_text(
                "config/early_stopping",
                f"monitor=fg_mDice, patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}",
                0,
            )

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "loss/train",
                float(self.logger.get_value("train_losses", step=-1)),
                self.current_epoch,
            )
            self._tb_writer.add_scalar(
                "learning_rate",
                float(self.optimizer.param_groups[0]["lr"]),
                self.current_epoch,
            )

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)
        tp = np.sum(outputs["tp_hard"], 0)
        fp = np.sum(outputs["fp_hard"], 0)
        fn = np.sum(outputs["fn_hard"], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            gathered_tp = [None] * world_size
            gathered_fp = [None] * world_size
            gathered_fn = [None] * world_size
            gathered_loss = [None] * world_size
            dist.all_gather_object(gathered_tp, tp)
            dist.all_gather_object(gathered_fp, fp)
            dist.all_gather_object(gathered_fn, fn)
            dist.all_gather_object(gathered_loss, outputs["loss"])
            tp = np.vstack(gathered_tp).sum(0)
            fp = np.vstack(gathered_fp).sum(0)
            fn = np.vstack(gathered_fn).sum(0)
            val_loss = float(np.vstack(gathered_loss).mean())
        else:
            val_loss = float(np.mean(outputs["loss"]))

        with np.errstate(divide="ignore", invalid="ignore"):
            dice = 2 * tp / (2 * tp + fp + fn)
            iou = tp / (tp + fp + fn)
        fg_mdice = float(np.nanmean(dice))
        fg_miou = float(np.nanmean(iou))

        self.logger.log("mean_fg_dice", fg_mdice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dice.tolist(), self.current_epoch)
        self.logger.log("val_losses", val_loss, self.current_epoch)

        metrics = {"fg_mDice": fg_mdice, "fg_mIoU": fg_miou}
        for name, class_dice, class_iou in zip(
            self._class_names(len(dice)), dice, iou
        ):
            metrics[f"Dice/{name}"] = float(class_dice)
            metrics[f"IoU/{name}"] = float(class_iou)

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss/validation", val_loss, self.current_epoch)
            for metric_name, value in metrics.items():
                self._tb_writer.add_scalar(
                    f"validation/{metric_name}", value, self.current_epoch
                )
                previous_best = self._best_metrics.get(metric_name, -np.inf)
                if np.isfinite(value):
                    self._best_metrics[metric_name] = max(previous_best, value)
                self._tb_writer.add_scalar(
                    f"best/{metric_name}",
                    self._best_metrics.get(metric_name, previous_best),
                    self.current_epoch,
                )
            self._tb_writer.flush()

        improved = fg_mdice > (
            self._best_fg_mdice + self.early_stopping_min_delta
        )
        if improved:
            self._best_fg_mdice = fg_mdice
            self._epochs_without_improvement = 0
            self.print_to_log_file(
                f"New best fg_mDice: {self._best_fg_mdice:.4f}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
        else:
            self._epochs_without_improvement += 1
            self.print_to_log_file(
                "fg_mDice did not improve: "
                f"{self._epochs_without_improvement}/"
                f"{self.early_stopping_patience}"
            )
            self._early_stop_requested = (
                self._epochs_without_improvement
                >= self.early_stopping_patience
            )

    def on_epoch_end(self):
        # Keep timing, periodic checkpoints and plots, but best checkpointing is
        # handled above using raw fg_mDice rather than nnU-Net's EMA.
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.get_value("train_losses", step=-1), 4),
        )
        self.print_to_log_file(
            "val_loss",
            np.round(self.logger.get_value("val_losses", step=-1), 4),
        )
        self.print_to_log_file(
            "Dice",
            np.round(
                self.logger.get_value("dice_per_class_or_region", step=-1), 4
            ).tolist(),
        )
        if (
            (self.current_epoch + 1) % self.save_every == 0
            and self.current_epoch != self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    def run_training(self):
        self.on_train_start()
        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [
                self.train_step(next(self.dataloader_train))
                for _ in range(self.num_iterations_per_epoch)
            ]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [
                    self.validation_step(next(self.dataloader_val))
                    for _ in range(self.num_val_iterations_per_epoch)
                ]
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
            if self._early_stop_requested:
                self.print_to_log_file(
                    "Early stopping: fg_mDice has not improved by at least "
                    f"{self.early_stopping_min_delta} for "
                    f"{self.early_stopping_patience} epochs."
                )
                break
        self.on_train_end()

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            if self._tb_writer is not None:
                self._tb_writer.close()

