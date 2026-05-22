class IterationDataPoint:
    COL_NAME_METRIC = "metric"
    COL_NAME_ITER_IDX = "idx_iter"
    COL_NAME_EPOCH_IDX = "idx_epoch"

    COL_NAMES = [
        COL_NAME_EPOCH_IDX
        , COL_NAME_ITER_IDX
        , COL_NAME_METRIC
    ]

    def __init__(
        self
        , iter_idx
        , epoch_idx
        , metric
    ):
        self.data = {}

        self.data[IterationDataPoint.COL_NAME_EPOCH_IDX] = epoch_idx
        self.data[IterationDataPoint.COL_NAME_ITER_IDX] = iter_idx
        self.data[IterationDataPoint.COL_NAME_METRIC] = metric

    def __str__(self):
        return "IterationDataPoint:\n\t+ epoch_idx: {epoch_idx}\n\t+ iter_idx: {iter_idx}\n\t+ metric: {metric}\n" \
            .format(
                epoch_idx=self.data[IterationDataPoint.COL_NAME_EPOCH_IDX]
                , iter_idx=self.data[IterationDataPoint.COL_NAME_ITER_IDX]
                , metric=self.data[IterationDataPoint.COL_NAME_METRIC]
            )