import torch
import numpy as np
import torch_geometric as pyg

from tqdm import tqdm
from torch import nn, optim
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    optim               : str
    n_epochs            : int
    weight_decay        : float
    learning_rate       : float
    train_loader        : DataLoader
    snapshot_interval   : Optional[int]                 = None
    val_loader          : Optional[DataLoader]          = None
    snapshot_x          : Optional[Tuple[np.ndarray]]   = None
    
    def __str__(self):
        return f"Train=[epochs={self.n_epochs}, optim={self.optim}, lr={self.learning_rate:1.0e}, weight_decay={self.weight_decay:1.0e}]"

@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    iter_idx        : int
    epoch_idx       : int
    batch_idx       : int  
    train_loss      : float
    train_error     : float
    val_loss        : Optional[float]                           = None
    val_error       : Optional[float]                           = None
    snapshot_y_hat  : Optional[Tuple[np.ndarray, np.ndarray]]   = None

class NNModel():
    def __init__(self, net: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        
        self.net = net.to(self.device)
        self.loss = nn.CrossEntropyLoss().to(self.device)

    def train(self, params: NNTrainParams):
        train_str = f"{self.net} x {params}"
        validate = params.val_loader is not None
        snapshot = params.snapshot_x is not None

        if params.optim == "sgd":
            optimizer = optim.SGD(
                lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        elif params.optim == "adam":
            optimizer = optim.Adam(
                lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        
        iter_idx: int = 0
        idps    : List[IterationDataPoint] = []

        tqdm_bar = tqdm(
            desc=train_str
            , total=int(params.n_epochs * len(params.train_loader))
        )

        with torch.set_grad_enabled(True):
            for epoch_idx in range(params.n_epochs):
                for batch_idx, batch in enumerate(params.train_loader):
                    self.net.train()
                    self.net.zero_grad()
                        
                    X, Y, Y_hat = self._iter_fwd(batch)
                    
                    train_loss = self.loss(Y_hat, Y)          
                    train_error = 1 - (Y_hat.max(dim=1)[1] == Y).sum().item() / Y.size(0)

                    train_loss.backward()
                    optimizer.step()

                    if validate:
                        val_loss, val_error = self.evaluate(loader=params.val_loader)
                    else:
                        val_loss, val_error = None, None, None
                        
                    if snapshot and (
                        params.snapshot_interval is None or (
                            (params.snapshot_interval is not None) and (iter_idx % params.snapshot_interval == 0)
                        )
                    ):
                        snapshot_y_hat = self.predict(X=params.snapshot_x)
                    else:
                        snapshot_y_hat = None

                    idp = NNIterationDataPoint(
                        epoch_idx=epoch_idx
                        , iter_idx=iter_idx
                        , val_loss=val_loss
                        , batch_idx=batch_idx
                        , val_error=val_error
                        , train_error=train_error
                        , train_loss=float(train_loss)
                        , snapshot_y_hat=snapshot_y_hat
                    )

                    idps.append(idp)

                    iter_idx += 1
                    tqdm_bar.update(1)
                    tqdm_bar.set_postfix_str(f"error: {val_error if val_error is not None else train_error:.4f}")
        
        return train_str, np.array(idps)

    def evaluate(self, loader: DataLoader):
        self.net.eval()
        loss_vals, err_vals = [], []

        with torch.no_grad():
            for _, batch in enumerate(loader):
                X, Y, Y_hat = self._iter_fwd(batch)
                
                loss = self.loss(Y_hat, Y)              
                loss_vals.append(float(loss))

                error = 1 - (Y_hat.max(dim=1)[1] == Y).sum().item() / Y.size(0)
                err_vals.append(float(error))

        return (
            np.mean(loss_vals)
            , np.mean(err_vals)
        )
        
    def _iter_fwd(self, batch):
        X, Y = self.net.unpack_batch(batch)
        
        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)
        
        Y_hat = self.net(*X)
        
        return X, Y, Y_hat
    
    def predict(self, X: np.ndarray):
        if not isinstance(X, tuple):
            X = (X,)
        
        X = tuple(torch.from_numpy(x).to(self.device) for x in X)
        
        self.net.eval()
        with torch.no_grad():
            Y_hat = self.net(*X).detach().numpy()
            
            return Y_hat, Y_hat.argmax(axis=1)