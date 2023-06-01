from __future__ import annotations

import torch
import numpy as np

from enum import Enum
from tqdm import tqdm
from sklearn import metrics
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field, replace

class Device(Enum):
    CPU     = "cpu"
    MPS     = "mps"
    CUDA    = "cuda"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)

    def __call__(self):
        return torch.device(self.value)

    @staticmethod
    def get():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Device.MPS
        elif torch.cuda.is_available():
            return Device.CUDA
        else:
            return Device.CPU

class Optim(Enum):
    SGD             = "sgd"
    ADAM            = "adam"
    ADAM_AMSGRAD    = "adam_amsgrad"
    SGD_NESTEROV    = "sgd_nesterov"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
    
    def __call__(self, net: nn.Module, params: NNOptimParams):
        assert net is not None and params is not None
        
        match self:
            case Optim.SGD:
                return optim.SGD(
                    lr=params.lr_start
                    , params=net.parameters()
                    , momentum=params.momentum
                    , weight_decay=params.weight_decay
                )
            case Optim.ADAM:
                return optim.Adam(
                    lr=params.lr_start
                    , betas=params.momentum
                    , params=net.parameters()
                    , weight_decay=params.weight_decay
                )
            case Optim.ADAM_AMSGRAD:
                return optim.Adam(
                    amsgrad=True
                    , lr=params.lr_start
                    , betas=params.momentum
                    , params=net.parameters()
                    , weight_decay=params.weight_decay
                )
            case Optim.SGD_NESTEROV:
                return optim.SGD(
                    nesterov=True
                    , lr=params.lr_start
                    , params=net.parameters()
                    , momentum=params.momentum
                    , weight_decay=params.weight_decay
                )
    
class Loss(Enum):
    CROSS_ENTROPY           = "cross_entropy"
    MEAN_SQUARED_ERROR      = "mean_squared_error"
    BINARY_CROSS_ENTROPY    = "binary_cross_entropy"
    NEGATIVE_LOG_LIKELIHOOD = "negative_log_likelihood"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)

    def __call__(self):
        match self:
            case Loss.CROSS_ENTROPY             : return nn.CrossEntropyLoss()
            case Loss.MEAN_SQUARED_ERROR        : return nn.BCELoss()
            case Loss.BINARY_CROSS_ENTROPY      : return nn.MSELoss()
            case Loss.NEGATIVE_LOG_LIKELIHOOD   : return nn.NLLLoss()

@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    lr_start    : float
    weight_decay: float
    momentum    : Union[float, Tuple[float, float]]
    
    def __str__(self):
        return f"[lr_start={self.lr_start:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}]"

@dataclass(frozen=True, kw_only=True, slots=True)
class NNSchedulerParams:
    patience    : int
    factor      : float
    threshold   : float
    
    def __str__(self):
        return f"[patience={self.patience}, factor={self.factor:1.0e}, threshold={self.threshold:1.0e}]"
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    n_epochs            : int
    optim               : Optim                 = Optim.ADAM
    scheduler_params    : NNSchedulerParams     = NNSchedulerParams(patience=5, factor=9e-1, threshold=1e-4)
    optim_params        : NNOptimParams         = NNOptimParams(lr_start=9e-1, momentum=(0.9, 0.999), weight_decay=5e-4)
    
    train_loader        : DataLoader
    val_loader          : Optional[DataLoader]  = None
    
    snapshot_epoch_delta: Optional[int]         = None
    
    def is_valid(self):
        if self.optim == Optim.SGD or self.optim == Optim.SGD_NESTEROV:
            return isinstance(self.optim_params.momentum, float)
        elif self.optim == Optim.ADAM or self.optim == Optim.ADAM_AMSGRAD:
            return (
                isinstance(self.optim_params.momentum, tuple)
                and len(self.optim_params.momentum) == 2
                and all(isinstance(x, float) for x in self.optim_params.momentum)
            )
    
    def __str__(self):
        return f"Train=[n_epochs={self.n_epochs}, optim={self.optim}] x OptimParams={self.optim_params} x SchedulerParams={self.scheduler_params}"
        
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
    loss        : Optional[float]   = None
    error       : Optional[float]   = None
    
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
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNModelParams:
    loss  : Loss
    device: Device
    
    def __str__(self):
        return f"[device={self.device}, loss={self.loss}]"
    
    def is_valid(self):
        return (
            self.device is not None
            and self.loss is not None
        )

class NNModel():
    def __init__(
        self
        , net   : nn.Module
        , params: NNModelParams = NNModelParams(device=Device.CPU, loss=Loss.CROSS_ENTROPY)
    ):
        assert (
            net is not None
            and params is not None
            and params.is_valid()
        )
         
        self.params     = params
        
        self.device     = self.params.device()
        self.net        = net.to(self.device)
        self.loss_fn    = self.params.loss().to(self.device)

    def train(self, params: NNTrainParams):
        assert params is not None and params.is_valid()
        
        train_str   = f"{self.params} x {self.net} x {params}"
        validate    = params.val_loader is not None
        snapshot    = validate and params.snapshot_epoch_delta is not None
        
        if snapshot:
            snapshot_x, snapshot_y  = self.net.unpack_batch(next(iter(params.val_loader)))
            snapshot_x, snapshot_y  = tuple(x.numpy() for x in snapshot_x), snapshot_y.numpy()
            
        optimizer = params.optim(net=self.net, params=params.optim_params)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer
            , mode='min'
            , factor=params.scheduler_params.factor
            , patience=params.scheduler_params.patience
            , threshold=params.scheduler_params.threshold
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
                
                scheduler.step(val_edp.error if val_edp is not None else train_edp)
                
                tqdm_bar.set_postfix_str(
                    "error={error}, lr={lr}"
                        .format(
                            error=f"{val_edp.error if val_edp is not None else train_edp.error:.4f}"
                            , lr=f"{optimizer.param_groups[0]['lr']:.4f}"
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