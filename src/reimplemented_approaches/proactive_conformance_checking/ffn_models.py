"""
Reimplementaiton of FFN seperate, collective for deviation prediction:
Grohs, M., Pfeiffer, P., Rehse, J.: Proactive conformance checking: An approach for predicting deviations in business processes. Inf. Syst. 127, 102461 (2025)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def _to_2d_float(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is shaped [batch, features] and float.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim > 2:
        x = x.view(x.size(0), -1)
    return x.float()


def _concat_features(x_act: torch.Tensor,
                     x_res: Optional[torch.Tensor] = None,
                     x_month: Optional[torch.Tensor] = None,
                     x_trace: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Concatenate feature blocks as described in the paper's FFN diagrams.
    Supports two calling conventions:
    - forward(x_concat): pass only x_act and leave others as None.
    - forward(x_act, x_res, x_month, x_trace): pass four feature blocks.
    """
    if x_res is None and x_month is None and x_trace is None:
        return _to_2d_float(x_act)
    if x_res is None or x_month is None or x_trace is None:
        raise ValueError("Provide either x_concat only OR all of x_act/x_res/x_month/x_trace.")

    x_act_f = _to_2d_float(x_act)
    x_res_f = _to_2d_float(x_res)
    x_month_f = _to_2d_float(x_month)
    x_trace_f = _to_2d_float(x_trace)
    return torch.cat([x_act_f, x_res_f, x_month_f, x_trace_f], dim=-1)

class FFNCollectiveIDP(nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_hidden_1: int,
                 fc_hidden_2: int,
                 num_output_labels: int = None,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")):
        
        super().__init__()
        if num_output_labels is None:
            raise ValueError("num_output_labels must be provided")
        self.device = torch.device(device)

        self.fc_hidden_1 = nn.Linear(input_size, fc_hidden_1)
        self.layer_norm_1 = nn.LayerNorm(fc_hidden_1)
        self.leaky_relu_1 = nn.LeakyReLU()

        self.fc_hidden_2 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.layer_norm_2 = nn.LayerNorm(fc_hidden_2)
        self.leaky_relu_2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc_output = nn.Linear(fc_hidden_2, num_output_labels)
        
        # kwargs important to save the model
        self.init_kwargs = dict(input_size=input_size,
                                fc_hidden_1=fc_hidden_1,
                                fc_hidden_2=fc_hidden_2,
                                num_output_labels=num_output_labels,
                                dropout=dropout,
                                device=self.device.type)
        
        self.to(self.device)

    def forward(self,
                x_act: torch.Tensor,
                x_res: Optional[torch.Tensor] = None,
                x_month: Optional[torch.Tensor] = None,
                x_trace: Optional[torch.Tensor] = None,
                apply_sigmoid: bool = False) -> torch.Tensor:

        x_concat = _concat_features(x_act, x_res, x_month, x_trace).to(self.device)

        x = self.fc_hidden_1(x_concat)
        x = self.layer_norm_1(x)
        x = self.leaky_relu_1(x)

        x = self.fc_hidden_2(x)
        x = self.layer_norm_2(x)
        x = self.leaky_relu_2(x)

        x = self.dropout(x)

        logits = self.fc_output(x)

        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
    
    def save(self, path: str):
        """
        Store the trained model at path.
        """
        checkpoint = {"model_state_dict": self.state_dict(),
                      "kwargs": self.init_kwargs}
        torch.save(checkpoint, Path(path))

    @staticmethod
    def load(path: str,
             device: Optional[torch.device] = None) -> "FFNCollectiveIDP":
        """
        Load the stored model at path.
        """
        checkpoint = torch.load(Path(path), weights_only=False, map_location=device or torch.device("cpu"))
        kwargs = checkpoint["kwargs"]
        if device is not None:
            kwargs["device"] = device

        model = FFNCollectiveIDP(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.device)
        model.eval()
        return model
    
class _SingleLabelIDP(nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_hidden_1: int,
                 fc_hidden_2: int,
                 dropout: float):
        
        super().__init__()
        
        self.fc_hidden_1 = nn.Linear(input_size, fc_hidden_1)
        self.layer_norm_1 = nn.LayerNorm(fc_hidden_1)
        self.leaky_relu_1 = nn.LeakyReLU()

        self.fc_hidden_2 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.layer_norm_2 = nn.LayerNorm(fc_hidden_2)
        self.leaky_relu_2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)
        self.fc_output = nn.Linear(fc_hidden_2, 1)

    def forward(self,
                x_concat: torch.Tensor) -> torch.Tensor:

        x = self.fc_hidden_1(x_concat)
        x = self.layer_norm_1(x)
        x = self.leaky_relu_1(x)

        x = self.fc_hidden_2(x)
        x = self.layer_norm_2(x)
        x = self.leaky_relu_2(x)

        x = self.dropout(x)
        return self.fc_output(x).squeeze(-1)

class FFNSeparateIDP(nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_hidden_1: int,
                 fc_hidden_2: int,
                 fc_out: int,
                 num_output_labels: int,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")):
        
        super().__init__()
        if num_output_labels is None or num_output_labels < 1:
            raise ValueError("num_output_labels must be provided and > 0")

        # Mirrors LSTMSeparateIDP: in this repo, "separate" uses cross-entropy with
        # logits of shape [batch, 2] per deviation label-model.
        if num_output_labels != 2:
            raise ValueError(
                "FFNSeparateIDP (in this codebase) is used as a binary classifier. "
                "num_output_labels must be 2 (class logits)."
            )

        self.device = torch.device(device)

        self.num_output_labels = num_output_labels

        # Two scalar-logit heads stacked into a 2-class logit vector.
        self.label_heads = nn.ModuleList([_SingleLabelIDP(input_size=input_size,
                                                          fc_hidden_1=fc_hidden_1,
                                                          fc_hidden_2=fc_hidden_2,
                                                          dropout=dropout) for _ in range(num_output_labels)])

        self.init_kwargs = dict(input_size=input_size,
                                fc_hidden_1=fc_hidden_1,
                                fc_hidden_2=fc_hidden_2,
                                fc_out=fc_out,
                                num_output_labels=num_output_labels,
                                dropout=dropout,
                                device=self.device.type)

        self.to(self.device)

    def forward(self,
                x_act: torch.Tensor,
                x_res: Optional[torch.Tensor] = None,
                x_month: Optional[torch.Tensor] = None,
                x_trace: Optional[torch.Tensor] = None,
                apply_softmax: bool = False) -> torch.Tensor:

        x_concat = _concat_features(x_act, x_res, x_month, x_trace).to(self.device)
        logits = torch.stack([head(x_concat) for head in self.label_heads], dim=-1)

        if apply_softmax:
            return torch.softmax(logits, dim=-1)
        return logits

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": self.init_kwargs}
        
        torch.save(checkpoint, Path(path))

    @staticmethod
    def load(path: str,
             device: Optional[torch.device] = None) -> "FFNSeparateIDP":
        checkpoint = torch.load(Path(path), weights_only=False, map_location=device or torch.device("cpu"))
        kwargs = checkpoint["kwargs"]
        if device is not None:
            kwargs["device"] = device
        model = FFNSeparateIDP(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.device)
        model.eval()
        return model
