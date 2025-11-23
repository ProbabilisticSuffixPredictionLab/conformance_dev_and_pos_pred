import math
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.counter = 0

    def __call__(self, model: torch.nn.Module, val_loss: float) -> bool:
        if self.best_loss is None or (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False


class Training:
    def __init__(self,
                 model: torch.nn.Module,
                 train_set: TensorDataset,
                 optimizer_values: Dict[str, Any],
                 device: torch.device = torch.device("cuda"),
                 saving_path: str = "./model.pkl",
                 label_weights: Optional[torch.Tensor] = None,
                 val_set: Optional[TensorDataset] = None,
                 early_stopping: Optional[Dict[str, Any]] = None):
        
        self.model = model
        self.train_set = train_set
        self.val_set = val_set

        self.batch_size = optimizer_values["mini_batches"]
        self.shuffle = optimizer_values["shuffle"]
        self.num_epochs = optimizer_values["epochs"]
        self.optimizer = optimizer_values["optimizer"]

        self.device = torch.device(device)
        self.saving_path = Path(saving_path)

        self.label_weights = (label_weights if label_weights is not None else self._compute_label_weights(train_set))
        self.label_weights = self.label_weights.to(self.device)

        self.early_stopper = None
        if early_stopping and self.val_set is not None and len(self.val_set) > 0:
            self.early_stopper = EarlyStopping(**early_stopping)

    def _compute_label_weights(self, data_source) -> torch.Tensor:
        if hasattr(data_source, "tensors"):
            dataset = data_source
        elif hasattr(data_source, "dataset") and hasattr(data_source.dataset, "tensors"):
            dataset = data_source.dataset
        else:
            raise ValueError("Training data must expose a TensorDataset to compute label weights.")

        targets = dataset.tensors[-1].detach().float()
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        num_samples = targets.shape[0]
        deviating = targets.sum(dim=0)
        conforming = num_samples - deviating

        lir = (conforming + 1e-8) / (deviating + 1e-8)
        denom = 2 * math.e + torch.log(lir)
        denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
        beta = (1.0 / 6.0) * (1.0 / denom)
        return torch.clamp(beta, min=1e-6)

    def _weighted_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.label_weights.to(preds.device)
        weights = weights.unsqueeze(0).expand_as(preds)
        return F.binary_cross_entropy_with_logits(preds, targets.float(), weight=weights)

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0, pin_memory=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
        for batch in progress:
            x_act, x_res, x_month, x_trace, y = batch
            x_act = x_act.to(self.device, non_blocking=True)
            x_res = x_res.to(self.device, non_blocking=True)
            x_month = x_month.to(self.device, non_blocking=True)
            x_trace = x_trace.to(self.device, non_blocking=True)
            
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(x_act, x_res, x_month, x_trace)
            loss = self._weighted_loss(logits, y)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item() * x_act.size(0)
            running_loss += batch_loss
            # progress.set_postfix(loss=loss.item())

        return running_loss / len(self.train_set)

    def _evaluate_loss(self, dataset: Optional[TensorDataset]) -> Optional[float]:
        if dataset is None or len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                x_act, x_res, x_month, x_trace, y = batch
                x_act = x_act.to(self.device, non_blocking=True)
                x_res = x_res.to(self.device, non_blocking=True)
                x_month = x_month.to(self.device, non_blocking=True)
                x_trace = x_trace.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.model(x_act, x_res, x_month, x_trace)
                loss = self._weighted_loss(logits, y)
                total_loss += loss.item() * x_act.size(0)

        return total_loss / len(dataset)

    def train(self) -> List[Dict[str, Any]]:
        self.model.to(self.device)
        history: List[Dict[str, Any]] = []

        for epoch in range(self.num_epochs):
            train_loss = self._run_epoch(epoch)
            val_loss = self._evaluate_loss(self.val_set)

            record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            history.append(record)
            tqdm.write(f"Epoch {epoch+1}/{self.num_epochs} — train loss: {train_loss:.4f}" +
                       (f", val loss: {val_loss:.4f}" if val_loss is not None else ""))

            if self.early_stopper and val_loss is not None:
                if self.early_stopper(self.model, val_loss):
                    tqdm.write("Early stopping triggered.")
                    break

        self.model.save(path=self.saving_path)
        return history