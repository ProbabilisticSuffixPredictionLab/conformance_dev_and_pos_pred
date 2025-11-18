import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Training:
    def __init__(self,
                 model: torch.nn.Module,
                 train_set: TensorDataset,
                 optimizer_values: Dict[str, Any],
                 device: torch.device = torch.device("cuda"),
                 saving_path: str = "./model.pkl",
                 label_weights: Optional[torch.Tensor] = None):
        
        self.model = model
        
        self.train_set = train_set

        self.batch_size = optimizer_values["mini_batches"]
        self.shuffle = optimizer_values["shuffle"]
        self.num_epochs = optimizer_values["epochs"]
        self.optimizer = optimizer_values["optimizer"]

        self.device = torch.device(device)
        self.saving_path = Path(saving_path)

        self.label_weights = (label_weights if label_weights is not None else self._compute_label_weights(train_set))
        self.label_weights = self.label_weights.to(self.device)

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

    def train(self) -> List[Dict[str, Any]]:
        self.model.to(self.device)
        history: List[Dict[str, Any]] = []

        for epoch in range(self.num_epochs):
            train_loss = self._run_epoch(epoch)
            
            self.model.save(path=self.saving_path)

            record = {"epoch": epoch + 1, "train_loss": train_loss}
            history.append(record)
            tqdm.write(f"Epoch {epoch+1}/{self.num_epochs} — train loss: {train_loss:.4f}")

        return history