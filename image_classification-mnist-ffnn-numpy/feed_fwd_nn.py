import numpy as np

from tqdm import tqdm
from utils import Utils
from consts import Consts
from relu_layer import ReluLayer
from linear_layer import LinearLayer
from iteration_data_point import IterationDataPoint
from softmax_cross_entropy_layer import SoftmaxCrossEntropyLayer

class FeedFwdNN:
    def __init__(
        self
        , idx_val           : np.ndarray
        , X_val             : np.ndarray
        , Y_val             : np.ndarray
        , idx_train         : np.ndarray
        , X_train           : np.ndarray
        , Y_train           : np.ndarray
        , lr                : float         = Consts.LR
        , n_epochs          : int           = Consts.N_EPOCHS
        , mini_batch_size   : int           = Consts.MINI_BATCH_SIZE
    ):
        assert idx_train is not None
        assert idx_train.ndim == 1

        assert X_train is not None
        assert X_train.ndim == 2
        
        assert Y_train is not None
        assert Y_train.ndim == 2

        assert idx_train.shape[0] <= X_train.shape[0]
        assert X_train.shape[0] == Y_train.shape[0]

        assert idx_val is not None
        assert idx_val.ndim == 1

        assert X_val is not None
        assert X_val.ndim == 2
        
        assert Y_val is not None
        assert Y_val.ndim == 2

        assert idx_val.shape[0] <= X_val.shape[0]
        assert X_val.shape[0] == Y_val.shape[0]

        assert X_train.shape[1] == X_val.shape[1]
        assert Y_train.shape[1] == Y_val.shape[1]

        self.I_train = idx_train
        self.X_train = X_train
        self.Y_train = Y_train

        self.I_val = idx_val
        self.X_val = X_val
        self.Y_val = Y_val

        self.lr = lr
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        self.feature_size_in = self.X_train.shape[1]
        self.feature_size_out = self.Y_train.shape[1]

        self.L1 = LinearLayer(feature_size_in=self.feature_size_in, feature_size_out=self.feature_size_out)
        self.L2 = ReluLayer(feature_size=self.feature_size_out)
        self.L3 = SoftmaxCrossEntropyLayer(feature_size=self.feature_size_out)

        print(f"NeuralNet.__init__:\n\t+ epochs: {self.n_epochs}\n\t+ mini_batch_size: {self.mini_batch_size}\n\t+ lr: {self.lr}\n")

    def train_and_validate(self):
        iter_idx: int = 0
        iteration_data = []
        
        tqdm_bar = tqdm(total=int(self.n_epochs * len(self.I_train) // self.mini_batch_size))

        for epoch_idx in range(self.n_epochs):
            for mb_idx, I in enumerate(Utils.mini_batchify(self.I_train, self.mini_batch_size)):
                X = self.X_train[I]
                Y = self.Y_train[I]

                A1 = self.L1.forward(X)
                A2 = self.L2.forward(A1)
                L_train = self.L3.forward(A2, Y)

                dA2 = self.L3.backward()
                dA1 = self.L2.backward(dA2)
                dW, db = self.L1.backward(dA1)

                self.L1.W -= self.lr * dW
                self.L1.b -= self.lr * db

                L_val = self._validate()

                idp = IterationDataPoint(
                    epoch_idx=epoch_idx
                    , mini_batch_idx=mb_idx
                    , iter_idx=iter_idx
                    , training_loss=L_train
                    , validation_loss=L_val
                )

                iteration_data.append(idp)

                iter_idx += 1
                tqdm_bar.update(1)

        return np.array(iteration_data)

    def _validate(self):
        loss_vals = []

        L1_val = LinearLayer(
            W=self.L1.W
            , b=self.L1.b
            , feature_size_in=self.feature_size_in
            , feature_size_out=self.feature_size_out
        )
        L2_val = ReluLayer(feature_size=self.feature_size_out)
        L3_val = SoftmaxCrossEntropyLayer(feature_size=self.feature_size_out)

        for I in Utils.mini_batchify(self.I_val, self.mini_batch_size):
            X = self.X_val[I]
            Y = self.Y_val[I]

            A1 = L1_val.forward(X)
            A2 = L2_val.forward(A1)
            L_val = L3_val.forward(A2, Y)

            loss_vals.append(L_val)
        
        return np.mean(loss_vals)