"""
Utilities for fixing things that go strangley awry in the databricks environment.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from pathlib import Path
from typing import List


def bulk_file_rename(dir: Path, char_to_replace: str = '_', prefered_char: str = '-') -> List[str]:
    """Lists the directory, renaming every file by replacing the givien `char_to_replace` with `prefered_char`
    Returns the new paths for the renaming files."""
    new_names = []
    for p in dir.iterdir():

        new_path = Path(p.parent, p.name.replace(char_to_replace, prefered_char))
        p.rename(new_path)
        new_names.append(new_path)
    return new_names
