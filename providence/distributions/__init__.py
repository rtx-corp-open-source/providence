# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    @staticmethod
    @abstractmethod
    def hazard(self):
        pass

    @staticmethod
    @abstractmethod
    def cumulative_hazard(self):
        pass
