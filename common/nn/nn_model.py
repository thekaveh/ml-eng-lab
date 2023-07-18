from __future__ import annotations

import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from dataclasses import asdict
from typing import List, Optional
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .enum.loss import Loss
from .enum.device import Device
from .enum.checkpoint_type import CheckpointType

from ..utils import Utils
from .params.nn_run import NNRun
from .params.nn_checkpoint import NNCheckpoint
from .params.nn_train_params import NNTrainParams
from .params.nn_model_params import NNModelParams
from .params.nn_iteration_data_point import NNIterationDataPoint
from .params.nn_evaluation_data_point import NNEvaluationDataPoint

class NNModel():
    def __init__(
        self
        , net   : nn.Module
        , params: NNModelParams = NNModelParams(device=Device.CPU, loss=Loss.CROSS_ENTROPY)
    ):
        assert net is not None
         
        self.params     = params
        
        self.device     = self.params.device()
        self.net        = net.to(self.device)
        self.loss_fn    = self.params.loss().to(self.device)

    def train(self, params: NNTrainParams) -> NNRun:
        assert (
            params is not None
            and params.optim_params is not None
            and params.optim_params.is_valid()
        )

        validate    : bool  = params.val_loader is not None
        run         : NNRun = NNRun(
            train_params=params
            , model_params=self.params
            , net_params=self.net.params
        )
            
        optimizer   = params.optim_params.optim(
            net=self.net
            , lr_start=params.optim_params.lr_start
            , momentum=params.optim_params.momentum
            , weight_decay=params.optim_params.weight_decay
        )

        scheduler   = lr_scheduler.ReduceLROnPlateau(
            optimizer
            , mode='min'
            , factor=params.scheduler_params.factor
            , patience=params.scheduler_params.patience
            , threshold=params.scheduler_params.threshold
        )
        
        idx_iter        : int                           = 0
        idps            : List[NNIterationDataPoint]    = []
        n_iter          : int                           = int(params.n_epochs * len(params.train_loader))
        best_checkpoint : Optional[NNCheckpoint]        = NNCheckpoint.load(run=run.id, type=CheckpointType.BEST)
        
        Utils.print_tree(run.rep)

        with (
            torch.set_grad_enabled(True)
            , tqdm(
                colour="blue"
                , total=n_iter
                , desc="[+] Training"
            ) as tqdm_bar
        ):
            for idx_epoch in range(params.n_epochs):
                for idx_batch, batch in enumerate(params.train_loader):
                    self.net.train()
                    self.net.zero_grad()
                        
                    X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                    
                    train_loss = self.loss_fn(Y_hat_log, Y)
                    
                    train_edp = (
                        NNEvaluationDataPoint.of(Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy())
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

                val_edp     = self.evaluate(loader=params.val_loader) if validate else None           
                idps[-1]    = idps[-1].with_val_edp(val_edp)           
                checkpoint  = NNCheckpoint(
                    idp=idps[-1]
                    , model_state=self.net.state_dict()
                    , optim_state=optimizer.state_dict()
                )
                
                if idx_epoch == 0:
                    checkpoint.save(run=run.id, type=CheckpointType.FIRST)     
                elif idx_epoch == int(params.n_epochs / 4) - 1:
                    checkpoint.save(run=run.id, type=CheckpointType.Q1)     
                elif idx_epoch == int(params.n_epochs / 2) - 1:
                    checkpoint.save(run=run.id, type=CheckpointType.Q2)  
                elif idx_epoch == int(3 * params.n_epochs / 4) - 1:
                    checkpoint.save(run=run.id, type=CheckpointType.Q3)
                    
                checkpoint.save(run=run.id, type=CheckpointType.LAST)
                
                if best_checkpoint is None or checkpoint.idp.val_edp.error < best_checkpoint.idp.val_edp.error:
                    best_checkpoint = checkpoint
                    checkpoint.save(run=run.id, type=CheckpointType.BEST)
                
                scheduler.step(val_edp.error if val_edp is not None else train_edp)
                
                tqdm_bar.set_postfix_str(
                    "error={error}, lr={lr}"
                        .format(
                            error=f"{val_edp.error if val_edp is not None else train_edp.error:.4f}"
                            , lr=f"{optimizer.param_groups[0]['lr']:.4f}"
                        )
                )
      
        print()
        return run.with_idps(idps).save()

    def evaluate(self, loader: DataLoader):
        self.net.eval()
        edps = []

        with torch.no_grad():
            for batch in loader:
                _, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                
                edps.append(
                    NNEvaluationDataPoint.of(Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy())
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