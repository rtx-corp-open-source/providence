# -*- coding: utf-8 -*-
import pytest
from providence.utils import datasets


@pytest.fixture
def nasa_train():
    return datasets.NasaDataSet(train=True)


@pytest.fixture
def nasa_test():
    return datasets.NasaDataSet(train=False)
