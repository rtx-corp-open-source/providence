"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.datasets.adapters import BackblazeQuarter, load_backblaze_csv


def test_backblaze_download():
    # download_backblaze_dataset(BackblazeQuarter._2020_Q1, data_root='.test-data-root')
    # NOTE: load_backblaze_csv attempts the download if it's not present
    df = load_backblaze_csv(BackblazeQuarter._2019_Q4, data_root='.test-data-root')
    assert df.shape
    print(df.shape)