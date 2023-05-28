from __future__ import annotations

import torch
import numpy as np
import torch_geometric as pyg

from enum import Enum
from tqdm import tqdm
from sklearn import metrics
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field, replace

class Optim(Enum):
    SGD             = "sgd"
    ADAM            = "adam"
    ADAM_AMSGRAD    = "adam_amsgrad"
    SGD_NESTEROV    = "sgd_nesterov"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
    
class Loss(Enum):
    CROSS_ENTROPY           = "ce"
    MEAN_SQUARED_ERROR      = "mse"
    BINARY_CROSS_ENTROPY    = "bce"
    NEGATIVE_LOG_LIKELIHOOD = "nll"
    
    @staticmethod
    def to_loss_fn(value: Loss):
        if value == Loss.CROSS_ENTROPY:
            return nn.CrossEntropyLoss()
        elif value == Loss.BINARY_CROSS_ENTROPY:
            return nn.BCELoss()
        elif value == Loss.MEAN_SQUARED_ERROR:
            return nn.MSELoss()
        elif value == Loss.NEGATIVE_LOG_LIKELIHOOD:
            return nn.NLLLoss()
        else:
            return nn.CrossEntropyLoss()

@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    n_epochs            : int
    weight_decay        : float
    learning_rate       : float
    momentum            : Union[float, Tuple[float, float]]
    
    scheduler_patience  : int                   = 5
    scheduler_factor    : float                 = 0.9
    scheduler_threshold : float                 = 1e-4
    
    train_loader        : DataLoader
    val_loader          : Optional[DataLoader]  = None
    
    snapshot_epoch_delta: Optional[int]         = None
    
    optim               : Optim                 = Optim.ADAM
    
    def is_valid(self):
        if optim == Optim.SGD or optim == Optim.SGD_NESTEROV:
            return isinstance(self.momentum, float)
        elif optim == Optim.ADAM or optim == Optim.ADAM_AMSGRAD:
            return (
                isinstance(self.momentum, tuple)
                and len(self.momentum) == 2
                and all(isinstance(x, float) for x in self.momentum)
            )
    
    def __str__(self):
        return f"Train=[n_epochs={self.n_epochs}, optim={self.optim}, lr={self.learning_rate:1.0e}, weight_decay={self.weight_decay:1.0e}]"
        
@dataclass(frozen=True, kw_only=True, slots=True)
class NNSnapshotDataPoint:
    y           : np.ndarray
    y_hat       : np.ndarray
    y_hat_log   : np.ndarray
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNEvaluationDataPoint:
    f1          : float
    recall      : float
    accuracy    : float
    precision   : float
    loss        : Optional[float]               = None
    error       : Optional[float]               = None
    
    def with_loss(self, value: float):
        return replace(self, loss=value)
    
    def with_error(self, value: float):
        return replace(self, error=value)
    
    @staticmethod
    def of(Y: np.ndarray, Y_hat: np.ndarray):
        return NNEvaluationDataPoint(
            accuracy=metrics.accuracy_score(y_true=Y, y_pred=Y_hat)
            , f1=metrics.f1_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
            , recall=metrics.recall_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
            , precision=metrics.precision_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
        )
    
    @staticmethod
    def mean_of(edps: List['NNEvaluationDataPoint']):
        ret = NNEvaluationDataPoint(
            f1=np.mean([edp.f1 for edp in edps])
            , recall=np.mean([edp.recall for edp in edps])
            , accuracy=np.mean([edp.accuracy for edp in edps])
            , precision=np.mean([edp.precision for edp in edps])
        )

        if len([edp.loss for edp in edps if edp.loss is not None]) > 0:
            ret = ret.with_loss(np.mean([edp.loss for edp in edps if edp.loss is not None]))
            
        if len([edp.error for edp in edps if edp.error is not None]) > 0:
            ret = ret.with_error(np.mean([edp.error for edp in edps if edp.error is not None]))

        return ret

@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    iter_idx        : int
    epoch_idx       : int
    batch_idx       : int  
    train_edp       : NNEvaluationDataPoint
    val_edp         : Optional[NNEvaluationDataPoint]   = None
    val_snapshot    : Optional[NNSnapshotDataPoint]     = None
    
    def with_val_edp(self, value: NNEvaluationDataPoint):
        return replace(self, val_edp=value)
    
    def with_val_snapshot(self, value: NNSnapshotDataPoint):
        return replace(self, val_snapshot=value)

class NNModel():
    def __init__(
        self
        , net   : nn.Module
        , device: str       = "cpu"
        , loss  : Loss      = Loss.CROSS_ENTROPY
    ):
        self.device     = torch.device(device)
        
        self.net        = net.to(self.device)
        self.loss_fn    = Loss.to_loss_fn(loss).to(self.device)

    def train(self, params: NNTrainParams):
        assert params is not None and params.is_valid()
        
        train_str   = f"{self.net} x {params}"
        validate    = params.val_loader is not None
        snapshot    = validate and params.snapshot_epoch_delta is not None
        
        if snapshot:
            snapshot_x, snapshot_y  = self.net.unpack_batch(next(iter(params.val_loader)))
            snapshot_x, snapshot_y  = tuple(x.numpy() for x in snapshot_x), snapshot_y.numpy()

        if params.optim == Optim.SGD:
            optimizer = optim.SGD(
                momentum=params.momentum
                , lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        elif params.optim == Optim.ADAM:
            optimizer = optim.Adam(
                lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        elif params.optim == Optim.ADAM_AMSGRAD:
            optimizer = optim.Adam(
                amsgrad=True
                , lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        elif params.optim == Optim.SGD_NESTEROV:
            optimizer = optim.SGD(
                nesterov=True
                , lr=params.learning_rate
                , momentum=params.momentum
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer
            , mode='min'
            , factor=params.scheduler_factor
            , patience=params.scheduler_patience
            , threshold=params.scheduler_threshold
        )
        
        idx_iter: int                           = 0
        idps    : List[NNIterationDataPoint]    = []
        n_iter  : int                           = int(params.n_epochs * len(params.train_loader))

        tqdm_bar = tqdm(
            desc=train_str
            , total=n_iter
        )

        with torch.set_grad_enabled(True):
            for idx_epoch in range(params.n_epochs):
                for idx_batch, batch in enumerate(params.train_loader):
                    self.net.train()
                    self.net.zero_grad()
                        
                    X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                    
                    train_loss = self.loss_fn(Y_hat_log, Y)
                    
                    train_edp = (
                        NNEvaluationDataPoint.of(Y=Y.numpy(), Y_hat=Y_hat.numpy())
                            .with_loss(value=float(train_loss))
                            .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
                    )

                    idps.append(
                        NNIterationDataPoint(
                            iter_idx=idx_iter
                            , epoch_idx=idx_epoch
                            , batch_idx=idx_batch
                            , train_edp=train_edp
                        )
                    )

                    train_loss.backward()
                    optimizer.step()

                    idx_iter += 1
                    tqdm_bar.update(1)
                    # tqdm_bar.set_postfix_str(
                    #     "train_error: {train_error} - val_error: ? - lr: {lr}"
                    #         .format(
                    #             train_error=f"{train_edp.error:.4f}"
                    #             , lr=str(optimizer.param_groups[0]['lr'])
                    #         )
                    # )

                val_edp = self.evaluate(loader=params.val_loader) if validate else None
                        
                if snapshot and (idx_epoch + 1) % params.snapshot_epoch_delta == 0:
                    snapshot_y_hat_log, snapshot_y_hat = self.predict(X=snapshot_x)
                    
                    val_snapshot = NNSnapshotDataPoint(
                        y=snapshot_y
                        , y_hat=snapshot_y_hat
                        , y_hat_log=snapshot_y_hat_log
                    )
                else:
                    val_snapshot = None
                    
                idps[-1] = (
                    idps[-1]
                        .with_val_edp(val_edp)
                        .with_val_snapshot(val_snapshot)
                )
                
                scheduler.step(val_edp.error)
                
                tqdm_bar.set_postfix_str(
                    "train_error: {train_error} - val_error: {val_error} - lr: {lr}"
                        .format(
                            train_error=f"{train_edp.error:.4f}"
                            , val_error=f"{val_edp.error:.4f}" if val_edp is not None else "-"
                            , lr=str(optimizer.param_groups[0]['lr'])
                        )
                )
        
        return train_str, np.array(idps)

    def evaluate(self, loader: DataLoader):
        self.net.eval()
        edps = []

        with torch.no_grad():
            for batch in loader:
                _, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                
                edps.append(
                    NNEvaluationDataPoint.of(Y=Y.numpy(), Y_hat=Y_hat.numpy())
                        .with_loss(value=float(self.loss_fn(Y_hat_log, Y)))
                        .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
                )

        return NNEvaluationDataPoint.mean_of(edps)

    def __fwd_pass(self, batch):
        X, Y = self.net.unpack_batch(batch)
        
        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)
        
        Y_hat_log = self.net(*X)
        Y_hat = Y_hat_log.argmax(dim=1)
        
        return X, Y, Y_hat_log, Y_hat
    
    def predict(self, X: np.ndarray):
        if not isinstance(X, tuple):
            X = (X,)
        
        X = tuple(torch.from_numpy(x).to(self.device) for x in X)
        
        self.net.eval()
        with torch.no_grad():
            Y_hat_log = self.net(*X).detach().numpy()
            Y_hat = Y_hat_log.argmax(axis=1)
            
            return Y_hat_log, Y_hat