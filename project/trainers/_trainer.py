from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
