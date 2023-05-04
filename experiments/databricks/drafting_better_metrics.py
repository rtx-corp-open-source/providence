"""
Purpose:
We have had an increasingly intense suspicion that our metrics are deceptive in aggregating the performance of
eventful entities with that of uneventful entities.
Our resolve of this issue is to be worked out in this file

Overview / Behavior herein:
Load a model in Databricks
Load the data for a given dataset
Test the more granual metrics

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from pathlib import Path

from providence.datasets import BackblazeDataset
from providence.datasets import DataSubsetId
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import concat_dataframes
from providence.datasets.adapters import load_nasa_dataframe
from providence.datasets.adapters import NasaTurbofanTest
from providence.paper_reproductions import GranularMetrics
from providence.paper_reproductions import NasaTransformer
from providence.training import LossAggregates
from providence.training import use_gpu_if_available
from providence.utils import also

# import torch as pt

# persisted_model = pt.load(
#     "/dbfs/FileStore/AIML/scratch/Providence-Attention-Axis/outputs/"
#     "experiment_1/2022-12-05T21:27:29/feature/backblaze/0th-best-Backblaze/checkpoints/"
#     "ProvidenceTransformer-epoch030-checkpoint-full.pt",
#     map_location='cpu',
# )

PROVIDENCE_DATA_ROOT = "/dbfs/FileStore/datasets/providence"
# ds = BackblazeDataset(
#     DataSubsetId.Test, quarter=BackblazeQuarter._2019_Q4, data_dir=PROVIDENCE_DATA_ROOT
# )
# ds.device = use_gpu_if_available()
# ds.use_device_for_iteration(True)

# # persisted_model.to(ds.device)

# fake_losses = LossAggregates([0.2, 0.1], [0.3, 0.2])
# metrics = GranualMetrics(persisted_model, ds, losses=fake_losses)
# print(metrics.T.sort_index().to_markdown())

################################################################################
#
# Assessing NASA Turbofan test set size proportions
#
################################################################################

dbfs_out_path = also(
    Path("/dbfs/FileStore/AIML/scratch/Providence-NASA-stats/"),
    lambda p: p.mkdir(parents=True, exist_ok=True),
)

for split_name in "train test".split():
    print(
        """
################################################################################
#
# Assessing NASA Turbofan sizes: {}
#
################################################################################
    """.format(
            split_name.upper()
        )
    )

    for current_nasa_test in NasaTurbofanTest.all():
        fdooX_df = load_nasa_dataframe(current_nasa_test, split_name=split_name, data_root=PROVIDENCE_DATA_ROOT)

        min_max_cycle_rul_by_unit_number = fdooX_df.groupby("unit number")[["RUL"]].agg(["max", "min"])

        count_steps_by_unit_number = fdooX_df.groupby("unit number")["cycle"].count().rename("available_timesteps")
        concat_dataframes(
            (min_max_cycle_rul_by_unit_number, count_steps_by_unit_number),
            axis="columns",
        ).to_csv(dbfs_out_path / f"{current_nasa_test}_{split_name}-itemized.csv")

        rul_desc = fdooX_df["RUL"].describe()
        rul_desc.to_csv(dbfs_out_path / f"{current_nasa_test}_{split_name}-RUL.describe.csv")

        cycle_desc = fdooX_df["cycle"].describe()
        cycle_desc.to_csv(dbfs_out_path / f"{current_nasa_test}_{split_name}-cycle.describe.csv")

        joint_desc = concat_dataframes((rul_desc, cycle_desc), axis="columns")
        print(f"{current_nasa_test}")
        print(joint_desc.to_markdown())
