import torch
import numpy as np

from tqdm import tqdm
from typing import List
from torch import nn, optim
from torch.utils.data import DataLoader

from consts import Consts
from feed_fwd_nn_module import FeedFwdNNModule
from iteration_data_point import IterationDataPoint

class FeedFwdNNModel():
    def __init__(
        self
        , loader_val    : DataLoader
        , loader_train  : DataLoader
        , lr            : float         = Consts.LR
        , device_name   : str           = Consts.DEVICE
        , n_epochs      : int           = Consts.N_EPOCHS
        , dropout_p     : float         = Consts.DROPOUT_P
        , optimizer_alg : str           = Consts.OPTIMIZER_ALG
        , input_size    : int           = Consts.FEATURES_SIZE_IN
        , output_size   : int           = Consts.FEATURES_SIZE_OUT
        , hidden_sizes  : List[int]     = Consts.FEATURES_SIZES_HIDDEN
    ):
        self.etha           = lr
        self.n_epochs       = n_epochs
        self.dropout_p      = dropout_p
        self.loader_val     = loader_val
        self.input_size     = input_size
        self.output_size    = output_size
        self.device_name    = device_name
        self.loader_train   = loader_train
        self.hidden_sizes   = hidden_sizes
        self.optimizer_alg  = optimizer_alg
        self.device         = torch.device(device_name)

        self.module = FeedFwdNNModule(
            dropout_prob    = self.dropout_p
            , input_size    = self.input_size
            , output_size   = self.output_size
            , hidden_sizes  = self.hidden_sizes
        ).to(self.device)
        
        self.smce = nn.CrossEntropyLoss().to(self.device)

        if optimizer_alg == "sgd":
            self.optimizer = optim.SGD(
                lr=self.etha
                , params=self.module.parameters()
            )
        elif optimizer_alg == "adam":
            self.optimizer = optim.Adam(
                lr=self.etha
                , params=self.module.parameters()
            )

    def train_and_validate(self):
        print(self)
        
        iter_idx: int = 0
        iteration_data = []

        tqdm_bar = tqdm(total=int(self.n_epochs * len(self.loader_train.batch_sampler)))

        for epoch_idx in range(self.n_epochs):
            for mb_idx, (X, Y) in enumerate(self.loader_train):
                self.module.zero_grad()

                X = X.to(self.device)
                Y = Y.to(self.device)

                Y_hat = self.module(X)
                L_train = self.smce(Y_hat, Y)

                total_preds = 0
                correct_preds = 0
                for Y_hat_idx, y_hat in enumerate(Y_hat):
                    total_preds += 1
                    if torch.argmax(y_hat) == Y[Y_hat_idx]:
                        correct_preds += 1
                E_train = 1 - correct_preds / total_preds

                L_train.backward()
                self.optimizer.step()

                L_val, E_val = self._validate()

                idp = IterationDataPoint(
                    epoch_idx=epoch_idx
                    , iter_idx=iter_idx
                    , mini_batch_idx=mb_idx
                    , training_error=E_train
                    , validation_error=E_val
                    , training_loss=float(L_train)
                    , validation_loss=float(L_val)
                )

                iteration_data.append(idp)

                iter_idx += 1
                tqdm_bar.update(1)
        
        return np.array(iteration_data)

    def _validate(self):
        loss_vals, err_vals = [], []

        with torch.no_grad():
            for _, (X, Y) in enumerate(self.loader_val):
                X = X.to(self.device)
                Y = Y.to(self.device)

                Y_hat = self.module(X)
                L_val = self.smce(Y_hat, Y)
                
                loss_vals.append(L_val)

                total_preds = 0
                correct_preds = 0
                for Y_hat_idx, y_hat in enumerate(Y_hat):
                    total_preds += 1
                    if torch.argmax(y_hat) == Y[Y_hat_idx]:
                        correct_preds += 1
                E_val = 1 - correct_preds / total_preds

                err_vals.append(E_val)

        return (
            torch.mean(torch.FloatTensor(loss_vals)).item()
            , torch.mean(torch.FloatTensor(err_vals)).item()
        )
    
    def __str__(self):
        return f"model={{sizes={[self.input_size] + self.hidden_sizes + [self.output_size]}, device={self.device_name}, epochs={self.n_epochs}, lr={self.etha}, optim={self.optimizer_alg}, dropout={self.dropout_p}}}"