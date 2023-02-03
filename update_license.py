#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
**Raytheon Technologies Proprietary**
Export Controlled - See license file

**Purpose**
This hook ensures the files are properly marked
"""
import subprocess
import time
from collections import namedtuple
from os import linesep


LICENSE_FILENAME = "LICENSE_AND_REQUIRED_MARKING.rst"


def git_ls_tree(dir="."):
    """Get a list of all files git is managing."""
    process = subprocess.Popen(
        ["git", "ls-tree", "--full-tree", "-r", "--name-only", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return namedtuple("Std", "out, err")(process.stdout.read(), process.stderr.read())


def git_out_to_list(git_out):
    """Convert git output to a list of files"""
    list_of_files = git_out.decode("utf-8").splitlines()
    return list_of_files


def replace_license(updated_file_list, filename=LICENSE_FILENAME):
    """Replace license file list of files with updated list."""
    # Read in file
    with open(filename, "r") as f:
        license_lines = f.read().splitlines()

    # Find line distinguishing the file list
    ear_idx = license_lines.index("EAR99 Files")

    # Remove all lines after
    license_lines = license_lines[: ear_idx + 2]

    # Append new file list
    license_lines += updated_file_list

    with open(filename, "w") as f:
        license_lines = f.write(linesep.join(license_lines))

    time.sleep(0.25)


def add_license(filename=LICENSE_FILENAME):
    """Add the license update to the commit."""
    _ = subprocess.Popen(["git", "add", filename], stdout=subprocess.PIPE)
    time.sleep(0.1)


def amend_commit(filename=LICENSE_FILENAME):
    """Amend the last commit with the license."""
    _ = subprocess.Popen(
        ["git", "commit", "--amend", "--no-edit", "--no-verify"], stdout=subprocess.PIPE
    )
    time.sleep(0.1)


def create_commit(filename=LICENSE_FILENAME):
    """Create a new commit with the license."""
    _ = subprocess.Popen(
        [
            "git",
            "commit",
            f"-m Update {filename} to be IP/GT compliant",
            "--no-verify",
        ],
        stdout=subprocess.PIPE,
    )
    time.sleep(0.1)


def update_license():
    """Update license file"""
    # Get names of all files being managed by git
    out, err = git_ls_tree()
    file_list = git_out_to_list(out)
    replace_license(file_list)
    add_license()
    # create_commit()


if __name__ == "__main__":
    update_license()
