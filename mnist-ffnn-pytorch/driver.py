import numpy as np
import torchvision as thv

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from utils import Utils
from consts import Consts
from neural_net_training import NeuralNet

class Driver:
    @staticmethod
    def main():
        data_train = thv.datasets.MNIST(
            root="./data"
            , train=True
            , download=True
            , transform=thv.transforms.ToTensor())

        I_train = np.random.randint(
            low=0
            , high=data_train.data.shape[0]
            , size=int(data_train.data.shape[0] * Consts.DS_SAMPLING_RATIO))

        loader_train = DataLoader(
            dataset=data_train
            , batch_size=Consts.MINI_BATCH_SIZE
            , sampler=SubsetRandomSampler(I_train))

        data_val = thv.datasets.MNIST(
            root="./data"
            , train=False
            , download=True
            , transform=thv.transforms.ToTensor())

        I_val = np.arange(0, int(data_val.data.shape[0] * Consts.DS_SAMPLING_RATIO))

        loader_val = DataLoader(
            dataset=data_val
            , batch_size=Consts.MINI_BATCH_SIZE
            , sampler=SequentialSampler(I_val))

        net = NeuralNet(
            epochs=1
            , etha=0.1
            , dropout_p=0.25
            , optimizer_alg="sgd"
            , loader_val=loader_val
            , loader_train=loader_train)

        iteration_data = net.train_and_validate()

        Utils.two_line_plot(
            y1_legend="Training"
            , y_axis_label="Loss"
            , y2_legend="Validation"
            , x_axis_label="Iterations"
            , fig_size=(2 * 2 * 2 * 3, 2 * 2 * 3)
            , x=[tld.iter_idx for tld in iteration_data]
            , y1=[tld.training_loss for tld in iteration_data]
            , y2=[tld.validation_loss for tld in iteration_data]
            , title="Training & Validation Losses for epoch(s)={epochs}, #iter/epoch={iter_epoch}, etha={etha}, optimizer={optimizer}, and dropout_p={dropout_p}"
                    .format(
                        etha=net.etha
                        , epochs=net.epochs
                        , dropout_p=net.dropout_p
                        , optimizer=net.optimizer_alg
                        , iter_epoch=len(iteration_data) // net.epochs))

        Utils.two_line_plot(
            y1_legend="Training"
            , y_axis_label="Error"
            , y2_legend="Validation"
            , x_axis_label="Iterations"
            , fig_size=(2 * 2 * 2 * 3, 2 * 2 * 3)
            , x=[tld.iter_idx for tld in iteration_data]
            , y1=[tld.training_error for tld in iteration_data]
            , y2=[tld.validation_error for tld in iteration_data]
            , title="Training & Validation Errors for epoch(s)={epochs}, #iter/epoch={iter_epoch}, etha={etha}, optimizer={optimizer}, and dropout_p={dropout_p}"
                    .format(
                        etha=net.etha
                        , epochs=net.epochs
                        , dropout_p=net.dropout_p
                        , optimizer=net.optimizer_alg
                        , iter_epoch=len(iteration_data) // net.epochs))

def main():
    Driver.main()

if __name__ == "__main__":
    main()