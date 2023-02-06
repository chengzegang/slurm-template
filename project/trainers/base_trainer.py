from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    ...


class Trainer(BaseTrainer):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
