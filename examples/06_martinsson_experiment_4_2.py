"""
Martinsson worked against the CMAPSS dataset, used in the PHM08 Prognostics Data Challenge,
particularly focused on FD002 and FD004.

This example shows how to use the low-level data-loading utilities to follow the prescribed
treatment outlined in Section 4.2.1

Author's note:
- This is not necessarily performance-optimal on load. However, if the downloads are cached you should be okay :)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from pandas import concat

from providence.dataloaders import BasicDataloaders
from providence.datasets import DataSubsetId
from providence.datasets import NasaFD00XDataset
from providence.datasets import ProvidenceDataset
from providence.datasets import train_test_split
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.datasets.adapters import NasaTurbofanTest
from providence.nn import ProvidenceLSTM
from providence.paper_reproductions import NasaRnnOptimizer
from providence.training import generic_training
from providence.training import use_gpu_if_available

"We created our own random fold by choosing a subset of the CMAPSS data (trainFD002 and trainFD004)"
martinsson_ds = NasaFD00XDataset(NasaTurbofanTest.FD002, NasaTurbofanTest.FD004, subset_choice=DataSubsetId.Train)
print(f"{len(martinsson_ds) = }")


# going a little out of order here, to better mesh with the library
def truncate_sequences(ds: ProvidenceDataset) -> ProvidenceDataset:
    """None of the engines brake down before 128 steps. To limit training time this eventful period
    was removed and the longest sequence was then truncated to 254 steps."""
    new_data = []
    for id, df in ds.data:
        if df.shape[0] > 128:
            uneventful_truncated = df[128:]
            longest_truncated = uneventful_truncated[:254]
            new_data.append((id, longest_truncated))

    # concat the newly truncated data, reassigning the grouping field (which *should* be lost in the groupby)
    new_df_all = concat(map(lambda id_df: id_df[1].assign(**{ds.grouping_field: id_df[0]}), new_data))
    return ProvidenceDataset(
        new_df_all,
        grouping_field=ds.grouping_field,
        feature_columns=ds.feature_columns,
        tte_column=ds.tte_column,
        event_indicator_column=ds.event_indicator_column,
        device=ds.device,
    )


"This was then randomly split to ... validation- and 249 training-sequences"
# For reasons unknown, we have far more devices than cited in Martinsson
# nevertheless, we'll stick to his training set size
training_percentage = 249 / len(martinsson_ds)
martinsson_ds = truncate_sequences(martinsson_ds)
train_ds, test_ds = train_test_split(martinsson_ds, split_percentage=training_percentage, seed=1234)

"With batch-size 1 this means that ..."
dls = BasicDataloaders(train_ds, test_ds, batch_size=1)

"LSTM was 26 x 100 x 10 x 2 ... i.e. the hidden state size was 100"
model = ProvidenceLSTM(
    len(NASA_FEATURE_NAMES),
    hidden_size=100,
    num_layers=10,
    device=use_gpu_if_available(),
)

"The model was trained for 60k iterations as shown in figure 4.3"
optimizer = NasaRnnOptimizer(model)._replace(num_epochs=60_000)  # lots of epochs

losses = generic_training(model, optimizer, dls)
