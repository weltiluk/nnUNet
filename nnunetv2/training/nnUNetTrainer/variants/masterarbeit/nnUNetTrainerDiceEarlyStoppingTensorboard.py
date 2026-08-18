"""nnU-Net trainer with MMSeg-like validation tracking.

The standard nnU-Net loss is retained: Dice + cross entropy for label maps
(Dice + BCE for region training). Early stopping uses the EMA-smoothed
foreground mean Dice. The best checkpoint uses the highest raw validation
foreground mean Dice.
"""

import os
import random
from os.path import join
from time import time
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainerDiceEarlyStoppingTensorboard(nnUNetTrainer):
    """Track classwise Dice/IoU and stop on EMA validation ``fg_mDice``.

    Configuration is intentionally done through environment variables because
    nnU-Net trainer classes have a fixed constructor:

    ``NNUNET_EARLY_STOPPING_PATIENCE`` (default 15),
    ``NNUNET_EARLY_STOPPING_MIN_DELTA`` (default 0.001).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        configured_batch_size = os.getenv("NNUNET_BATCH_SIZE")
        if configured_batch_size is not None:
            configured_batch_size = int(configured_batch_size)
            if configured_batch_size < 1:
                raise ValueError("NNUNET_BATCH_SIZE must be >= 1")
            self.configuration_manager.configuration["batch_size"] = (
                configured_batch_size
            )
        self.early_stopping_patience = int(
            os.getenv("NNUNET_EARLY_STOPPING_PATIENCE", "15")
        )
        self.early_stopping_min_delta = float(
            os.getenv("NNUNET_EARLY_STOPPING_MIN_DELTA", "0.001")
        )
        if self.early_stopping_patience < 1:
            raise ValueError("NNUNET_EARLY_STOPPING_PATIENCE must be >= 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("NNUNET_EARLY_STOPPING_MIN_DELTA must be >= 0")

        self._best_fg_mdice = -np.inf
        self._best_ema_fg_mdice = -np.inf
        self._epochs_without_improvement = 0
        self._early_stop_requested = False
        self._tb_writer = None
        self._best_metrics = {}
        # TensorBoard uses optimizer updates as its x-axis. Checkpoints are
        # written at epoch boundaries, so this can be reconstructed on resume.
        self._optimizer_step = 0

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
        # Seed immediately before the base trainer initializes the network so
        # runs with the same architecture start from identical weights. We do
        # not force deterministic CUDA algorithms or data augmentation here,
        # because those settings would reduce training performance.
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        super().on_train_start()
        self._optimizer_step = self.current_epoch * self.num_iterations_per_epoch
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
                f"monitor=EMA(fg_mDice), "
                f"checkpoint_monitor=raw fg_mDice, "
                f"x_axis=optimizer_step, "
                f"validation_interval={self.num_iterations_per_epoch} steps, "
                f"patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}",
                0,
            )

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "loss/train",
                float(self.logger.get_value("train_losses", step=-1)),
                self._optimizer_step,
            )
            self._tb_writer.add_scalar(
                "learning_rate",
                float(self.optimizer.param_groups[0]["lr"]),
                self._optimizer_step,
            )

    def train_step(self, batch: dict) -> dict:
        output = super().train_step(batch)
        self._optimizer_step += 1
        return output

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)
        tp = np.sum(outputs["tp_hard"], 0)
        fp = np.sum(outputs["fp_hard"], 0)
        fn = np.sum(outputs["fn_hard"], 0)
        tp_all = np.sum(outputs["tp_hard_all"], 0)
        fp_all = np.sum(outputs["fp_hard_all"], 0)
        fn_all = np.sum(outputs["fn_hard_all"], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            gathered_tp = [None] * world_size
            gathered_fp = [None] * world_size
            gathered_fn = [None] * world_size
            gathered_loss = [None] * world_size
            gathered_tp_all = [None] * world_size
            gathered_fp_all = [None] * world_size
            gathered_fn_all = [None] * world_size
            dist.all_gather_object(gathered_tp, tp)
            dist.all_gather_object(gathered_fp, fp)
            dist.all_gather_object(gathered_fn, fn)
            dist.all_gather_object(gathered_tp_all, tp_all)
            dist.all_gather_object(gathered_fp_all, fp_all)
            dist.all_gather_object(gathered_fn_all, fn_all)
            dist.all_gather_object(gathered_loss, outputs["loss"])
            tp = np.vstack(gathered_tp).sum(0)
            fp = np.vstack(gathered_fp).sum(0)
            fn = np.vstack(gathered_fn).sum(0)
            tp_all = np.vstack(gathered_tp_all).sum(0)
            fp_all = np.vstack(gathered_fp_all).sum(0)
            fn_all = np.vstack(gathered_fn_all).sum(0)
            val_loss = float(np.vstack(gathered_loss).mean())
        else:
            val_loss = float(np.mean(outputs["loss"]))

        with np.errstate(divide="ignore", invalid="ignore"):
            dice = 2 * tp / (2 * tp + fp + fn)
            iou = tp / (tp + fp + fn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            fscore = 2 * precision * recall / (precision + recall)
            dice_all = 2 * tp_all / (2 * tp_all + fp_all + fn_all)
            iou_all = tp_all / (tp_all + fp_all + fn_all)
            recall_all = tp_all / (tp_all + fn_all)
            precision_all = tp_all / (tp_all + fp_all)
            fscore_all = 2 * precision_all * recall_all / (precision_all + recall_all)
        fg_mdice = float(np.nanmean(dice))
        fg_miou = float(np.nanmean(iou))

        self.logger.log("mean_fg_dice", fg_mdice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dice.tolist(), self.current_epoch)
        self.logger.log("val_losses", val_loss, self.current_epoch)
        ema_fg_mdice = float(
            self.logger.get_value("ema_fg_dice", step=-1)
        )

        metrics = {
            "Dice": float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())),
            "mDice": float(np.nanmean(dice_all)),
            "fg_mDice": fg_mdice,
            "EMA_fg_mDice": ema_fg_mdice,
            "IoU": float(tp.sum() / (tp.sum() + fp.sum() + fn.sum())),
            "mIoU": float(np.nanmean(iou_all)),
            "fg_mIoU": fg_miou,
            "Recall": float(tp.sum() / (tp.sum() + fn.sum())),
            "mRecall": float(np.nanmean(recall_all)),
            "fg_mRecall": float(np.nanmean(recall)),
            "Precision": float(tp.sum() / (tp.sum() + fp.sum())),
            "mPrecision": float(np.nanmean(precision_all)),
            "fg_mPrecision": float(np.nanmean(precision)),
            "Fscore": float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())),
            "mFscore": float(np.nanmean(fscore_all)),
            "fg_mFscore": float(np.nanmean(fscore)),
        }
        for (
            name,
            class_dice,
            class_iou,
            class_recall,
            class_precision,
            class_fscore,
        ) in zip(
            self._class_names(len(dice)),
            dice,
            iou,
            recall,
            precision,
            fscore,
        ):
            metrics[f"Dice/{name}"] = float(class_dice)
            metrics[f"IoU/{name}"] = float(class_iou)
            metrics[f"Recall/{name}"] = float(class_recall)
            metrics[f"Precision/{name}"] = float(class_precision)
            metrics[f"Fscore/{name}"] = float(class_fscore)

        if not self.label_manager.has_regions:
            metrics["Dice/background"] = float(dice_all[0])
            metrics["IoU/background"] = float(iou_all[0])
            metrics["Recall/background"] = float(recall_all[0])
            metrics["Precision/background"] = float(precision_all[0])
            metrics["Fscore/background"] = float(fscore_all[0])

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "loss/validation", val_loss, self._optimizer_step
            )
            for metric_name, value in metrics.items():
                self._tb_writer.add_scalar(
                    metric_name, value, self._optimizer_step
                )
                previous_best = self._best_metrics.get(metric_name, -np.inf)
                if np.isfinite(value):
                    self._best_metrics[metric_name] = max(previous_best, value)
                self._tb_writer.add_scalar(
                    f"best/{metric_name}",
                    self._best_metrics.get(metric_name, previous_best),
                    self._optimizer_step,
                )
            self._tb_writer.flush()

        # Model selection deliberately uses the raw validation fg_mDice.
        if np.isfinite(fg_mdice) and fg_mdice > self._best_fg_mdice:
            self._best_fg_mdice = fg_mdice
            self.print_to_log_file(
                "New best raw validation fg_mDice: "
                f"{self._best_fg_mdice:.4f}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        # Early stopping deliberately uses only the EMA-smoothed fg_mDice.
        ema_improved = ema_fg_mdice > (
            self._best_ema_fg_mdice + self.early_stopping_min_delta
        )
        if ema_improved:
            self._best_ema_fg_mdice = ema_fg_mdice
            self._epochs_without_improvement = 0
            self._early_stop_requested = False
            self.print_to_log_file(
                "New best EMA fg_mDice for early stopping: "
                f"{self._best_ema_fg_mdice:.4f}"
            )
        else:
            self._epochs_without_improvement += 1
            self.print_to_log_file(
                "EMA fg_mDice did not improve: "
                f"{self._epochs_without_improvement}/"
                f"{self.early_stopping_patience}"
            )
            self._early_stop_requested = (
                self._epochs_without_improvement
                >= self.early_stopping_patience
            )

    def load_checkpoint(self, filename_or_checkpoint):
        super().load_checkpoint(filename_or_checkpoint)

        raw_history = self.logger.get_value("mean_fg_dice", step=None)
        ema_history = self.logger.get_value("ema_fg_dice", step=None)

        finite_raw = [float(value) for value in raw_history if np.isfinite(value)]
        self._best_fg_mdice = max(finite_raw, default=-np.inf)

        self._best_ema_fg_mdice = -np.inf
        self._epochs_without_improvement = 0
        for value in ema_history:
            value = float(value)
            if not np.isfinite(value):
                continue
            if value > (
                self._best_ema_fg_mdice + self.early_stopping_min_delta
            ):
                self._best_ema_fg_mdice = value
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

        self._early_stop_requested = (
            self._epochs_without_improvement >= self.early_stopping_patience
        )

    def on_epoch_end(self):
        # Keep timing, periodic checkpoints and plots, but best checkpointing is
        # handled above using raw fg_mDice. EMA is only for early stopping.
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
                    "Early stopping: EMA fg_mDice has not improved by at least "
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

