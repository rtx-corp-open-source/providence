"""
Purpose: hold the hardcoded instantiations of the work that led to our paper.
This is a simplifying approach and will open up the API surface
for additional experimentation that is just a slight deviation from the existing code base.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from logging import getLogger
from typing import Dict
from typing import Tuple

from pandas import concat as concat_dataframes
from pandas import DataFrame
from torch.nn import Module as TorchModule
from torch.optim import SGD

from providence.datasets import ProvidenceDataset
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.distributions import Weibull
from providence.metrics import MetricsCalculator
from providence.nn import ProvidenceGRU
from providence.nn import ProvidenceModule
from providence.nn import ProvidenceRNN
from providence.nn.transformer import ProvidenceTransformer
from providence.training import generic_training
from providence.training import LossAggregates
from providence.training import OptimizerWrapper
from providence.types import DataLoaders

logger = getLogger(__file__)


################################################################################
# NASA Aggregate dataset
################################################################################
def NasaTraining(model: ProvidenceModule, optimizer: OptimizerWrapper, dataloaders: DataLoaders) -> LossAggregates:
    """One-line pass-through to ``generic_training`` for the NASA models.

    See ``training.generic_training`` for more.

    Args:
        model (ProvidenceModule): a ProvidenceModule that is also a PyTorch module
        optimizer (OptimizerWrapper): Optimizer and hyperparameters to use in the training
        dataloaders (DataLoaders): Data and dataloaders to use in the training. See ``dataloaders.NasaDataLoaders``

    Returns:
        LossAggregates: All training- and validation-phase losses from training ``model`` on the given ``dataloaders``
    """
    assert isinstance(model, TorchModule)
    loss_agg = generic_training(model, optimizer, dataloaders)
    return loss_agg


def NasaTransformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the NASA aggregate dataset.

    Returns:
        ProvidenceTransformer: the one we found most effective on this dataset
    """
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=128,
        n_layers=2,
        n_attention_heads=2,
        dropout=0.5,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=1000,
    )
    return model


def NasaTransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the NASA aggregate dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``NasaTransformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 3e-4)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=1000)


def NasaRNN() -> ProvidenceRNN:
    """The best RNN that performed on the NASA aggregate dataset.

    Returns:
        ProvidenceRNN: the ProvidenceGRU we found most effective on this dataset
    """
    return ProvidenceGRU(len(NASA_FEATURE_NAMES), hidden_size=128, num_layers=2, dropout=0.5)


def NasaRnnOptimizer(model: ProvidenceRNN) -> OptimizerWrapper:
    """Construct the optimizer attached to the RNN ``model`` that was best on the NASA aggregate dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceRNN): a ProvidenceRNN, preferrably ``NasaRNN()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    opt = SGD(model.parameters(), lr=3e-2)
    return OptimizerWrapper(opt, batch_size=256, num_epochs=1000)


################################################################################
# NASA FD001 dataset
################################################################################
NasaFD001Training = NasaTraining


def NasaFD001Transformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the NASA FD001 dataset.

    Returns:
        ProvidenceTransformer: the one we found most effective on this dataset
    """
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=512,
        n_layers=4,
        n_attention_heads=2,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700,
    )
    return model


def NasaFD001TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the NASA FD001 dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``NasaFD001Transformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=500)


################################################################################
# NASA FD002 dataset
################################################################################
NasaFD002Training = NasaTraining


def NasaFD002Transformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the NASA FD002 dataset.

    Returns:
        ProvidenceTransformer: the one we found most effective on this dataset
    """
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=1024,
        n_layers=4,
        n_attention_heads=3,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700,
    )
    return model


def NasaFD002TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the NASA FD002 dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``NasaFD002Transformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=32, num_epochs=200)


################################################################################
# NASA FD003 dataset
################################################################################
NasaFD003Training = NasaTraining


def NasaFD003Transformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the NASA FD003 dataset.

    Returns:
        ProvidenceTransformer: the one we found most effective on this dataset
    """
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=512,
        n_layers=4,
        n_attention_heads=4,
        dropout=0.1,
        layer_norm_epsilon=1e-3,
        positional_encoding_dimension=700,
    )
    return model


def NasaFD003TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the NASA FD003 dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``NasaFD003Transformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 3e-2)
    return OptimizerWrapper(model_optim, batch_size=32, num_epochs=50)


################################################################################
# NASA FD004 dataset
################################################################################
NasaFD004Training = NasaTraining


def NasaFD004Transformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the NASA FD004 dataset.

    Returns:
        ProvidenceTransformer: the one we found most effective on this dataset
    """
    n_features = len(NASA_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=128,
        n_layers=4,
        n_attention_heads=2,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700,
    )
    return model


def NasaFD004TransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the NASA FD004 dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``NasaFD004Transformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 1e-2)
    return OptimizerWrapper(model_optim, batch_size=16, num_epochs=100)


################################################################################
# Backblaze dataset
################################################################################
def BackblazeTransformerSeed() -> int:
    """The best seed we found best for initalization and training of the BackblazeTransformer and optimizer."""
    return 14682599077633313808  # Long int


def BackblazeTraining(model: ProvidenceModule, optimizer: OptimizerWrapper, dataloaders: DataLoaders) -> LossAggregates:
    assert isinstance(model, TorchModule)
    loss_agg = generic_training(model, optimizer, dataloaders)
    return loss_agg


def BackblazeTransformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the Backblaze dataset.

    Returns:
        ProvidenceTransformer: the one we found highly effective on this dataset
    """

    n_features = len(BACKBLAZE_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=64,
        n_layers=2,
        n_attention_heads=2,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=700,
    )
    return model


def BackblazeTransformerOptimizer(model: ProvidenceTransformer) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the Backblaze dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``BackblazeTransformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 1e-2)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=500)


def BackblazeRnnSeed() -> int:
    """The best seed we found best for initalization and training of the BackblazeRNN and optimizer."""
    # seed = 1123705791327780546, for sure.
    return 1123705791327780546


def BackblazeRNN() -> ProvidenceRNN:
    """The best RNN that performed on the Backblaze dataset.

    Returns:
        ProvidenceRNN: the one we found most effective on this dataset
    """
    n_features = len(BACKBLAZE_FEATURE_NAMES)
    return ProvidenceGRU(input_size=n_features, hidden_size=256, num_layers=4, dropout=0.9)


def BackblazeRnnOptimizer(model: ProvidenceRNN) -> OptimizerWrapper:
    """Construct the optimizer attached to the RNN ``model`` that was best on the Backblaze dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceRNN): a ProvidenceRNN, preferrably ``BackblazeRNN()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 3e-2)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=300)


################################################################################
# BackblazeExtended dataset
################################################################################
def BackblazeExtendedTransformer() -> ProvidenceTransformer:
    """The best Transformer that performed on the BackblazeExtended dataset.

    Returns:
        ProvidenceTransformer: the one we found highly effective on this dataset
    """
    n_features = len(BACKBLAZE_FEATURE_NAMES)
    model = ProvidenceTransformer(
        n_features,
        hidden_size=1024,
        n_layers=4,
        n_attention_heads=4,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        positional_encoding_dimension=1000,
    )
    return model


def BackblazeExtendedTransformerOptimizer(
    model: ProvidenceTransformer,
) -> OptimizerWrapper:
    """Construct the optimizer attached to the Transformer ``model`` that was best on the BackblazeExtended dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceTransformer): a ProvidenceTransformer, preferrably ``BackblazeExtendedTransformer()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """

    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=500)


def BackblazeExtendedRNN() -> ProvidenceRNN:
    """The best RNN that performed on the BackblazeExtended dataset.

    Returns:
        ProvidenceRNN: a Providence GRU we found highly effective on this dataset
    """
    return ProvidenceGRU(len(BACKBLAZE_FEATURE_NAMES), hidden_size=128, num_layers=4, dropout=0.1)


def BackblazeExtendedRnnOptimizer(model: ProvidenceRNN) -> OptimizerWrapper:
    """Construct the optimizer attached to the RNN ``model`` that was best on the BackblazeExtended dataset.

    This is the best optimizer configuration at the time of first writing the paper.

    Args:
        model (ProvidenceRNN): a ProvidenceRNN, preferrably ``BackblazeExtendedRNN()``

    Returns:
        OptimizerWrapper: tuple of the best SGD optimizer, best batch size, and optimal number of epochs
    """
    model_optim = SGD(model.parameters(), 3e-3)
    return OptimizerWrapper(model_optim, batch_size=128, num_epochs=300)


################################################################################
#
# General utilities
#
################################################################################

Metrics = Dict[str, float]


def compute_loss_metrics(losses: LossAggregates) -> Metrics:
    """Returns loss metrics for the ``lossess`` based on range statistics."""
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
    model: ProvidenceModule,
    ds: ProvidenceDataset,
    losses: LossAggregates,
    *,
    tte_cutoff: int = 150,
    all_on_cpu: bool = False,
    dist_t=Weibull,
) -> DataFrame:
    """Produce the ``fleet_metrics`` report for the fleet in ``ds``.

    Args:
        model (ProvidenceModule): model to generate metrics for
        ds (ProvidenceDataset): the dataset which holds the data for the ``model`` to be evaluated against
        losses (LossAggregates): the losses to produce loss-related metrics for, from the ``model``'s training
        tte_cutoff (int, optional): The time before which we are not concerned for metrics evaluation.
            See ``MetricsCalculator`` and environs for more. Defaults to 150.
        all_on_cpu (bool, optional): Whether to send the model and all data to the CPU before evaluting.
            This may be necessary in some environments, but we recommend against it. Evaluation can be blazingly fast
            if left to run on the GPU. Defaults to False.
        rul_stat (str, optional): Statistic to run on the output distribution, to be interpreted as a point-estimate.
            See ``MetricsCalculator`` and environs for more. Defaults to "mode".

    Returns:
        DataFrame: the output of ``providence.metrics::generate_metrics_table()`` plus the following metrics:
        - loss_{train|val}_
            - train or val is herein ``{run}``
                - total: sum of loss for the {run}
                - min: minimum loss achieved in the {run}
                - max: maximum loss achieved in the {run}
                - final: the last loss observed in the {run}

    TODO(stephen): add parameter for rul_stat
    """
    if all_on_cpu:
        model.to("cpu")
        ds.device = "cpu"  # NOTE potentially finicky.
    metrics_df = MetricsCalculator(model, dist_t, ds).fleet_metrics(tte_cutoff=tte_cutoff, rul_stat="mode")
    logger.info("computed metrics_df")

    logger.info("computing losses")
    loss_metrics = compute_loss_metrics(losses)
    metrics_df = metrics_df.assign(**loss_metrics)

    return metrics_df


def partition_ds(ds: ProvidenceDataset) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    """Partitions the Dataset by the event column.

    Behavior is only defined for entities that experience one event: a terminal event e.g. end of life of a deployed engine.
    Behavior is undefined (not yet fully reasoned through) for entities that experience the same event multiple times and continue
    to exist in the data set. Use with caution if this is your case.

    Args:
        ds (ProvidenceDataset): source dataset to split based on the uneventfulness of the entities therein.

    Returns:
        Tuple[ProvidenceDataset, ProvidenceDataset]: tuple (eventful_data, uneventful_data)
    """
    # probably don't need to df.assign here, but we're being cautious and using more guaranteed, slower behavior
    eventful_dfs, uneventful_dfs = [], []
    for entity_id, df in ds.iter_entities_with_id():
        df_identity_marked = df.assign(**{ds.grouping_field: entity_id})
        if df[ds.event_indicator_column].sum() > 0:  # has event
            eventful_dfs.append(df_identity_marked)
        else:
            uneventful_dfs.append(df_identity_marked)

    eventful_df = concat_dataframes(eventful_dfs)
    uneventful_df = concat_dataframes(uneventful_dfs)

    eventful_ds = ProvidenceDataset(
        eventful_df,
        grouping_field=ds.grouping_field,
        feature_columns=ds.feature_columns,
        tte_column=ds.tte_column,
        event_indicator_column=ds.event_indicator_column,
    )
    uneventful_ds = ProvidenceDataset(
        uneventful_df,
        grouping_field=ds.grouping_field,
        feature_columns=ds.feature_columns,
        tte_column=ds.tte_column,
        event_indicator_column=ds.event_indicator_column,
    )
    return eventful_ds, uneventful_ds


def GranularMetrics(
    model: ProvidenceModule,
    ds: ProvidenceDataset,
    losses: LossAggregates,
    *,
    tte_cutoff: int = 200,
    all_on_cpu: bool = False,
    suffix_eventful="eventful",
    suffix_uneventful="uneventful",
    rul_stat="mode",
) -> DataFrame:
    """Compute fleet-level metrics per partition of censoring condition of each item in ``ds``.

    After the computation, suffix the metric columns with ``suffix_eventful`` and ``suffix_uneventful``
    (delimited by underscore "_") parameters.

    Args:
        model (ProvidenceModule): model to generate metrics for
        ds (ProvidenceDataset): the dataset which holds the data for the ``model`` to be evaluated against
        losses (LossAggregates): the losses to produce loss-related metrics for, from the ``model``'s training
        tte_cutoff (int, optional): The time before which we are not concerned for metrics evaluation.
            See ``MetricsCalculator`` and environs for more. Defaults to 200.
        all_on_cpu (bool, optional): Whether to send the model and all data to the CPU before evaluting.
            This may be necessary in some environments, but we recommend against it. Evaluation can be blazingly fast
            if left to run on the GPU. Defaults to False.
        suffix_eventful (str, optional): The suffix to apply to the names given to the metrics generate on the eventful
            entities in the dataset. Defaults to "eventful".
        suffix_uneventful (str, optional): The suffix to apply to the names given to the metrics generate on uneventful
            entities in the dataset. Defaults to "uneventful".
        rul_stat (str, optional): Statistic to run on the output distribution, to be interpreted as a point-estimate.
            See ``MetricsCalculator`` and environs for more. Defaults to "mode".

    Returns:
        DataFrame: metrics evaluated twice - once on the eventful, once on the uneventful - excepting the loss metrics
            This is useful because it typically better to evaluate hard metrics against devices for which an event has
            occured, while for the uneventful we want to review forecasts. See ``GeneralMetrics`` for more.
            NOTE(stephen): there may be utility in checking the loss metrics in the "eventful split" view as well.
            TODO(stephen): extract dist_t inline with GeneralMetrics(...)
    """
    if all_on_cpu:
        model.to("cpu")
        ds.device = "cpu"  # NOTE potentially finicky.
    ds_eventful, ds_uneventful = partition_ds(ds)  # using ds.event_indicator_column, partition into two datasets
    metrics_eventful, metrics_uneventful = [
        MetricsCalculator(model, Weibull(), ds_sub).fleet_metrics(tte_cutoff=tte_cutoff, rul_stat=rul_stat)
        for ds_sub in (ds_eventful, ds_uneventful)
    ]

    logger.info("renaming metric dfs with their eventful suffixes")
    metrics_eventful.rename(lambda c: f"{c}_{suffix_eventful}", axis="columns", inplace=True)
    metrics_uneventful.rename(lambda c: f"{c}_{suffix_uneventful}", axis="columns", inplace=True)

    logger.info("computing loss metrics")
    loss_metrics = compute_loss_metrics(losses)
    metrics_df = concat_dataframes((metrics_eventful, metrics_uneventful), axis="columns").assign(**loss_metrics)

    return metrics_df
