class IterationDataPoint:
    def __init__(
        self
        , iter_idx: int
        , epoch_idx: int
        , mini_batch_idx: int
        , training_loss: float
        , training_error: float
        , validation_loss: float
        , validation_error: float
    ):
        self.iter_idx = iter_idx
        self.epoch_idx = epoch_idx
        self.training_loss = training_loss
        self.training_error = training_error
        self.mini_batch_idx = mini_batch_idx
        self.validation_loss = validation_loss
        self.validation_error = validation_error

    def __str__(self):
        return "IterationDataPoint:\n\t+ epoch_idx: {epoch_idx}\n\t+ mb_idx: {mini_batch_idx}\n\t+ iter_idx: {iter_idx}\n\t+ train_loss: {training_loss}\n\t+ train_error: {training_error}\n\t+ val_loss: {validation_loss}\n\t+ val_error: {validation_error}\n" \
            .format(
                epoch_idx=self.epoch_idx
                , mini_batch_idx=self.mini_batch_idx
                , iter_idx=self.iter_idx
                , training_loss=self.training_loss
                , training_error=self.training_error
                , validation_loss=self.validation_loss
                , validation_error=self.validation_error
            )