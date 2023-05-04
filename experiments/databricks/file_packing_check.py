"""
Purpose: demonstrate how to access files. Basically, just use DBFS after an upload.
I intended to use the packaged build (and the included JSON) files, but no pathing reference seems to be valid.

If you find a correction, please update this code and notify the team.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import json
from pathlib import Path


def relative_file(fpath: str) -> str:
    return (Path(__file__).parent / fpath).as_posix()


DBFS_CONFIG_HOME = "/dbfs/FileStore/AIML/scratch/Providence-Attention-Axis/configs"

# _EXPERIMENT_1_CONFIGS = json.load(open(f"{DBFS_CONFIG_HOME}/experiment-001.json"))["configurations"]

_EXPERIMENT_1_CONFIGS = list(
    filter(
        lambda cfg: cfg["name"].startswith("0th"),
        json.load(open(f"{DBFS_CONFIG_HOME}/experiment-001.json"))["configurations"],
    )
)

print(_EXPERIMENT_1_CONFIGS)
