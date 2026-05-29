class IterationDataPoint:
    def __init__(
        self
        , iter_idx: int
        , epoch_idx: int
        , mini_batch_idx: int
        , training_loss: float
        , validation_loss: float
    ):
        self.iter_idx           = iter_idx
        self.epoch_idx          = epoch_idx
        self.training_loss      = training_loss
        self.mini_batch_idx     = mini_batch_idx
        self.validation_loss    = validation_loss

    def __str__(self):
        return f"IterationDataPoint:\n\t+ epoch_idx: {self.epoch_idx}\n\t+ mb_idx: {self.mini_batch_idx}\n\t+ iter_idx: {self.iter_idx}\n\t+ train_loss: {self.training_loss}\n\t+ val_loss: {self.validation_loss}\n"
