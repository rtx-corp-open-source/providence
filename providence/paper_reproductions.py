"""
Purpose: hold the hardcoded instantiations of the work that led to our paper.
This is a simplifying approach and will open up the API surface
for additional experimentation that is just a slight deviation from the existing code base.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from logging import getLogger
from typing import Dict, Tuple

from pandas import DataFrame, concat as concat_dataframes

from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES, NASA_FEATURE_NAMES
from providence.distributions import Weibull
from providence.metrics import MetricsCalculator
from providence.datasets import ProvidenceDataset
from providence.nn import ProvidenceGRU, ProvidenceModule, ProvidenceRNN
from providence.nn.transformer import ProvidenceTransformer
from providence.training import LossAggregates, OptimizerWrapper, generic_training
from providence.types import DataLoaders
from torch.nn import Module as TorchModule
from torch.optim import SGD

logger = getLogger(__file__)


################################################################################
# NASA Aggregate dataset
################################################################################
def NasaTraining(model: ProvidenceModule, optimizer: OptimizerWrapper, dataloaders: DataLoaders) -> LossAggregates:
    assert isinstance(model, TorchModule)
    loss_agg = generic_training(model, optimizer, dataloaders)
    return loss_agg


def NasaTransformer() -> ProvidenceTransformer:
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=128,
        n_layers=2,
        n_attention_heads=2,
        dropout=0.5,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=1000
    )
    return model


def NasaTransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-4)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=1000)


def NasaRNN() -> ProvidenceRNN:
    return ProvidenceGRU(len(NASA_FEATURE_NAMES), hidden_size=128, num_layers=2, dropout=0.5)


def NasaRnnOptimizer(model: ProvidenceRNN) -> OptimizerWrapper:
    opt = SGD(model.parameters(), lr=3e-2)
    return OptimizerWrapper(opt, batch_size=256, num_epochs=1000)


################################################################################
# NASA FD001 dataset
################################################################################
NasaFD001Training = NasaTraining


def NasaFD001Transformer() -> ProvidenceTransformer:
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=512,
        n_layers=4,
        n_attention_heads=2,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700
    )
    return model


def NasaFD001TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=500)


################################################################################
# NASA FD002 dataset
################################################################################
NasaFD002Training = NasaTraining


def NasaFD002Transformer() -> ProvidenceTransformer:
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=1024,
        n_layers=4,
        n_attention_heads=3,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700
    )
    return model


def NasaFD002TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=32, num_epochs=200)


################################################################################
# NASA FD003 dataset
################################################################################
NasaFD003Training = NasaTraining


def NasaFD003Transformer() -> ProvidenceTransformer:
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=512,
        n_layers=4,
        n_attention_heads=4,
        dropout=0.1,
        layer_norm_epsilon=1e-3,
        positional_encoding_dimension=700
    )
    return model


def NasaFD003TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-2)
    return OptimizerWrapper(model_optim, batch_size=32, num_epochs=50)


################################################################################
# NASA FD004 dataset
################################################################################
NasaFD004Training = NasaTraining


def NasaFD004Transformer() -> ProvidenceTransformer:
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=128,
        n_layers=4,
        n_attention_heads=2,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700
    )
    return model


def NasaFD004TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 1e-2)
    return OptimizerWrapper(model_optim, batch_size=16, num_epochs=100)


################################################################################
# Backblaze dataset
################################################################################
def BackblazeTransformerSeed() -> int:
    return 14682599077633313808  # Long int


def BackblazeTraining(model: ProvidenceModule, optimizer: OptimizerWrapper, dataloaders: DataLoaders) -> LossAggregates:
    assert isinstance(model, TorchModule)
    loss_agg = generic_training(model, optimizer, dataloaders)
    return loss_agg


def BackblazeTransformer() -> ProvidenceTransformer:
    n_features = len(BACKBLAZE_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=64,
        n_layers=2,
        n_attention_heads=2,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700
    )
    return model


def BackblazeTransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 1e-2)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=500)


def BackblazeRnnSeed() -> int:
    # seed = 1123705791327780546, for sure.
    return 1123705791327780546


def BackblazeRNN() -> ProvidenceRNN:
    n_features = len(BACKBLAZE_FEATURE_NAMES)
    return ProvidenceGRU(input_size=n_features, hidden_size=256, num_layers=4, dropout=0.9)


def BackblazeRnnOptimizer(model: ProvidenceRNN) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-2)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=300)


################################################################################
# BackblazeExtended dataset
################################################################################
def BackblazeExtendedTransformer() -> ProvidenceTransformer:
    n_features = len(BACKBLAZE_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=1024,
        n_layers=4,
        n_attention_heads=4,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=1000
    )
    return model


def BackblazeExtendedTransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=500)


def BackblazeExtendedRNN() -> ProvidenceRNN:
    return ProvidenceGRU(len(BACKBLAZE_FEATURE_NAMES), hidden_size=128, num_layers=4, dropout=0.1)


def BackblazeExtendedRnnOptimizer(model: ProvidenceRNN) -> OptimizerWrapper:
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=300)


################################################################################
#
# General utilities
#
################################################################################

Metrics = Dict[str, float]


def compute_loss_metrics(losses: LossAggregates) -> Metrics:
    import numpy as np
    return {
        "loss_train_total": np.nansum(losses.training_losses),
        "loss_train_max": np.nanmax(losses.training_losses),
        "loss_train_min": np.nanmin(losses.training_losses),
        "loss_train_final": losses.training_losses[-1],
        "loss_val_total": np.nansum(losses.validation_losses),
        "loss_val_max": np.nanmax(losses.validation_losses),
        "loss_val_min": np.nanmin(losses.validation_losses),
        "loss_val_final": losses.validation_losses[-1],
        "final_epoch": min(len(losses.training_losses), len(losses.validation_losses)),
    }


def GeneralMetrics(
    model: ProvidenceModule, ds: ProvidenceDataset, losses: LossAggregates, *, tte_cutoff: int = 150, all_on_cpu: bool = False,
    dist_t = Weibull
) -> DataFrame:
    """
    Expect the following metrics in addition to those documented in `providence.metrics::generate_metrics_table()`:
    - loss_*
        - train & val (herein `{run}`)
            - total: sum of loss for the {run}
            - min: minimum loss achieved in the {run}
            - max: maximum loss achieved in the {run}
    """
    if all_on_cpu:
        model.to('cpu')
        ds.device = 'cpu' # NOTE potentially finicky.
    metrics_df = MetricsCalculator(model, dist_t, ds).fleet_metrics(tte_cutoff=tte_cutoff, rul_stat="mode")
    logger.info("computed metrics_df")

    logger.info("computing losses")
    loss_metrics = compute_loss_metrics(losses)
    metrics_df = metrics_df.assign(**loss_metrics)

    return metrics_df


def partition_ds(ds: ProvidenceDataset) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    """
    Partitions the Dataset by the event column.
    Behavior is only defined for entities that experience one event: a terminal event e.g. end of life of a deployed engine.
    Behavior is undefined (not yet fully reasoned through) for entities that experience the same event multiple times and continue
    to exist in the data set. Use with caution if this is your case.
    """
    # probably don't need to do `assign` bit, but we've seen variable behavior here below and are being cautious
    eventful_dfs, uneventful_dfs = [], []
    for entity_id, df in ds.iter_entities_with_id():
        df_identity_marked = df.assign(**{ds.grouping_field: entity_id})
        if df[ds.event_indicator_column].sum() > 0: # has event
            eventful_dfs.append(df_identity_marked)
        else:
            uneventful_dfs.append(df_identity_marked)
    
    eventful_df = concat_dataframes(eventful_dfs)
    uneventful_df = concat_dataframes(uneventful_dfs)

    eventful_ds = ProvidenceDataset(eventful_df, grouping_field=ds.grouping_field, feature_columns=ds.feature_columns, tte_column=ds.tte_column, event_indicator_column=ds.event_indicator_column)
    uneventful_ds = ProvidenceDataset(uneventful_df, grouping_field=ds.grouping_field, feature_columns=ds.feature_columns, tte_column=ds.tte_column, event_indicator_column=ds.event_indicator_column)
    return eventful_ds, uneventful_ds


def GranularMetrics(
    model: ProvidenceModule, ds: ProvidenceDataset, losses: LossAggregates, *, tte_cutoff: int = 200, all_on_cpu: bool = False,
    suffix_eventful = "eventful", suffix_uneventful = "uneventful", rul_stat="mode"
) -> DataFrame:
    """
    Compute fleet-level metrics per partition of censoring condition of each item in `ds`, suffixing the metric columns with `suffix_eventful`
    and `suffix_uneventful` (delimited by underscore '_') parameters.
    """
    if all_on_cpu:
        model.to('cpu')
        ds.device = 'cpu' # NOTE potentially finicky.
    ds_eventful, ds_uneventful = partition_ds(ds) # using ds.event_indicator_column, partition into two datasets
    metrics_eventful, metrics_uneventful = [
        MetricsCalculator(model, Weibull, ds_sub).fleet_metrics(tte_cutoff=tte_cutoff, rul_stat=rul_stat)
        for ds_sub in (ds_eventful, ds_uneventful)
    ]

    logger.info("renaming metric dfs with their eventful suffixes")
    metrics_eventful.rename(lambda c: f"{c}_{suffix_eventful}", axis="columns", inplace=True)
    metrics_uneventful.rename(lambda c: f"{c}_{suffix_uneventful}", axis="columns", inplace=True)

    logger.info("computing loss metrics")
    loss_metrics = compute_loss_metrics(losses)
    metrics_df = concat_dataframes((metrics_eventful, metrics_uneventful), axis='columns').assign(**loss_metrics)

    return metrics_df