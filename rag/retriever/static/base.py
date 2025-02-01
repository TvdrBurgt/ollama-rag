from abc import ABC, abstractmethod


class Ingestor(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def ingest(self, path: str):
        pass
