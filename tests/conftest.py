# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from glob import glob

def ensure_proxyoff() -> None:
    import os
    for proxy_key in "http_proxy https_proxy no_proxy".split():
        if proxy_key in os.environ:
            del os.environ[proxy_key]

def refactor(string: str) -> str:
    return string.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [refactor(fixture) for fixture in glob("tests/fixtures/*.py") if "__" not in fixture]

def pytest_configure(config):
    for marker in [
        "download_test: tests that run to make sure we can download the data we depend on",
        "transformers: mark each of the tests related to the transformer blocks"
    ]:
        config.addinivalue_line("markers", marker)
    ensure_proxyoff()
