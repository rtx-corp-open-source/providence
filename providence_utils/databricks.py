"""
Utilities for fixing things that go strangley awry in the databricks environment.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from os import PathLike
from pathlib import Path
from typing import Sequence


def bulk_file_rename(dir: Path, char_to_replace: str = "_", prefered_char: str = "-") -> Sequence[PathLike]:
    """Renames every file in ``dir`` by replacing the givien ``char_to_replace`` with ``prefered_char``.

    Created because Databricks filesystem would rename files with a hyphenated name to that of an underscore.
    As we thought this was an overrich for a generic file sytsem, we sought to remediate the issue

    Args:
        dir (Path): directory with files to be renamed. Does not discriminate, so produce with caution.
        char_to_replace (str, optional): The character to be replaced with ``preferred_char``. Defaults to "_".
        preferred_char (str, optional): The character with which to replace ``char_to_replace``. Defaults to "-".

    Returns:
        List[PathLike]: the new paths for the renamed files.
    """
    new_names = []
    for p in dir.iterdir():
        new_path = p.parent / p.name.replace(char_to_replace, prefered_char)
        p.rename(new_path)
        new_names.append(new_path)
    return new_names
