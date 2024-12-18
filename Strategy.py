from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def get_position(self, historical_data, current_position):
        pass

    def fit(self, data):
        pass