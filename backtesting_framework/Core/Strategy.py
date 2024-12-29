from abc import ABC, abstractmethod

class Strategy(ABC):

    def __init__(self, multi_asset=False):
        self.multi_asset = multi_asset

    @abstractmethod
    def get_position(self, historical_data, current_position):
        pass

    def fit(self, data):
        pass