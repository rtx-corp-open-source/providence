"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from enum import Enum, auto
from typing import Iterator, List, NamedTuple, Optional, Tuple

import torch as pt
from pandas import DataFrame
from torch import Tensor, from_numpy, ones_like
from torch import stack as stack_tensors
from torch import zeros_like
from torch.utils.data import Dataset


class ProvidenceDataset(Dataset):
    """
    A Dataset that presents a sequence as an item, designed for time-to-event prediction.
    It assumes the full sequence to be modeled (not necessarily the full lifetime of the device) is available in the
      supplied DataFrame.
    
    You can use this in conjunction with `assemble_trainable_by_entity_id()` and `compute_tte()` from `providence.dataset.adapters`
    to turn any (reasonable) dataset into a Providence-compatible dataset.

    Using just `compute_tte()`
    >>> df : DataFrame = load_my_data('device_1')
    >>> df.columns
    Index(['device_id', 'feature_1', 'feature_2', 'event_occurs', 'parsed_datetime'], dtype='object')
    >>> tte_column_name = 'tte'
    >>> df[tte_column_name] = compute_tte(df['parsed_datetime'])
    >>> # repeat for devices 2..n, assign 
    >>> df = pd.concat([df, df2, df3, ..., dfn])
    >>> ProvidenceDataset(df,
                        grouping_field='device_id',
                        feature_columns=['feature_1', 'feature_2'],
                        tte_column='tte',
                        event_indicator_column='event_occurs'
        )
    
    or, using `assemble_trainable_by_entity_id()`
    >>> df = load_all_my_data()
    >>> df.columns
    Index(['device_id', 'feature_1', 'feature_2', 'event_occurs', 'parsed_datetime'], dtype='object')
    >>> df = assemble_trainable_by_entity_id(df,
                                    entity_id='device_id',
                                    temporality_indicator='parsed_datetime',
                                    event_occurence_column='event_occurs')
    >>> ProvidenceDataset(df,
                        grouping_field='device_id',
                        feature_columns=['feature_1', 'feature_2'],
                        tte_column='tte',
                        event_indicator_column='event_occurs'
        )


    Arguments:
        df: source DataFrame that's been preprocessed
    
    Keyword Arguments:
        grouping_field: the column in df that identifies the underlying entity e.g. device, engine, person
        feature_columns: columns to extract for the group. Categorical features should be numericalized.
        tte_column: countdown to the event. Should terminate at 1 (not 0, as you might intuit).
            Default is `'tte'`
        event_indicator_column: the entire column should be a single bit, 1 or 0. This indicates where the event occurs at the
            end of the sequence (or not). Default is `'failure'`
    """
    def __init__(
        self,
        # coupling to the DataFrame because that's what we use. Could be more flexible, but at what benefit?
        df: DataFrame,
        *,
        grouping_field: str,
        feature_columns: List[str],
        tte_column: str = 'tte',
        event_indicator_column: str = 'failure',
        device: Optional[pt.device] = None
    ):
        self.feature_columns = feature_columns
        self.tte_column = tte_column
        self.event_indicator_column = event_indicator_column
        self.grouping_field = grouping_field
        self.grouped = df.groupby(grouping_field)
        self.data = list(self.grouped)
        self.device = device or pt.device('cpu')

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self._get_single_item(index)

    def _get_single_item(self, index):
        _, df = self.data[index]
        feature_np, tte_np = df[self.feature_columns].to_numpy(), df[self.tte_column].to_numpy()

        indicator_column = from_numpy(df[self.event_indicator_column].to_numpy())

        bit_tens = ones_like(indicator_column) if indicator_column.sum().to(bool) else zeros_like(indicator_column)

        tte_tens = from_numpy(tte_np)
        targets_tens = stack_tensors((tte_tens, bit_tens), dim=1)

        feature_tens = from_numpy(feature_np)
        return (  #yapf: skip
            feature_tens.to(pt.get_default_dtype()),
            targets_tens.to(pt.get_default_dtype()),
        )

    def __len__(self) -> int:
        return len(self.data)

    def use_device_for_iteration(self, should_use: bool) -> None:
        self._use_device = should_use

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def iter_entities_with_id(self) -> Iterator[Tuple[str, DataFrame]]:
        for i in range(len(self)):
            yield self.data[i]

    def iter_tensors_with_id(self) -> Iterator[Tuple[str, Tuple[Tensor, Tensor]]]:
        if getattr(self, '_use_device', None):
            for i in range(len(self)):
                yield self._get_id_by_index(i), tuple([tens.to(self.device) for tens in self._get_single_item(i)])
        else:
            for i in range(len(self)):
                yield self._get_id_by_index(i), self._get_single_item(i)

    def _get_id_by_index(self, index: int) -> str:
        return self.data[index][0]

    def get_device_by_index(self, index: int) -> DataFrame:
        return self.data[index][1]

    def get_device_by_id(self, id: str) -> DataFrame:
        return self.grouped.get_group(id)


################################################################################
#
# General utilities
# (Declared upfront because Python is a single-pass, late-bind interpreter)
################################################################################


class DataFrameSplit(NamedTuple):
    "Wrapper for the splits that we do all the time, just to clean things up."
    train: DataFrame
    test: DataFrame
    validation: Optional[DataFrame] = None


class DataSubsetId(Enum):
    Train = auto()
    Test = auto()
    Validation = auto()
