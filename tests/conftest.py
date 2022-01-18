# -*- coding: utf-8 -*-
from glob import glob


def refactor(string: str) -> str:
    return string.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [refactor(fixture) for fixture in glob("tests/fixtures/*.py") if "__" not in fixture]

def pytest_configure(config):
    config.addinivalue_line(
        "markers", """
        transformers: mark each of the tests related to the transformer blocks
        """
    )
